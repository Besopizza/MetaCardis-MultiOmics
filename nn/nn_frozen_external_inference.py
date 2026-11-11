import os, re, importlib.util, pickle
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('saved_meta_ensemble', os.path.join(HERE, 'nn_final_ensemble.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(os.path.join(HERE, 'final_outputs', 'best_overall_nn', 'deployment_models.pkl'), 'rb') as f:
    bundle = m.ModelUnpickler(f).load()
assert bundle['method'] == 'ensemble' and set(bundle['models']) == {('direct', 'micro'), ('direct', 'metabo'), ('impute', 'micro'), ('impute', 'metabo')}
models = bundle['models']
mi_cols = bundle['microbiome_features']
me_cols = bundle['metabolome_features']
ref = next(iter(models.values())).ref

def chemical_key(x):
    x = re.sub('^SERUM_ABS_', '', str(x))
    x = re.sub('^(C\\d+_|(?:HILIC|C18)-(?:neg|pos)_Cluster_\\d+:\\s*)', '', x)
    x = re.sub('_[12]$', '', x)
    x = re.sub('(?i)^(l|d|dl)_', '', x)
    return re.sub('[^a-z0-9]', '', x.lower())

def run(v, tag):
    mi = pd.read_csv(os.path.join(os.path.dirname(HERE), 'data', f'microbiome({v}).csv'), index_col=0)
    me = pd.read_csv(os.path.join(os.path.dirname(HERE), 'data', f'metabolome({v}).csv'), index_col=0)
    assert mi.index.is_unique and me.index.is_unique and mi.index.equals(me.index)
    hm = mi.notna().any(axis=1).to_numpy()
    he = me.notna().any(axis=1).to_numpy()
    assert (hm | he).all()
    ids = mi.index
    M = mi.reindex(columns=mi_cols, fill_value=0).fillna(0).to_numpy(dtype=float)
    keys = {}
    for c in me.columns:
        keys.setdefault(chemical_key(c), []).append(c)
    matches = {c: keys[chemical_key(c)][0] for c in me_cols if chemical_key(c) in keys}
    E = np.broadcast_to(ref.s_me.mean_, (len(ids), len(me_cols))).copy()
    for j, c in enumerate(me_cols):
        if c not in matches:
            continue
        source = matches[c]
        vals = me.loc[he, source].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5 or np.std(vals) < 1e-12:
            continue
        original = me[source].to_numpy(dtype=float)
        idx = np.flatnonzero(np.isfinite(original) & he)
        ranks = np.searchsorted(np.sort(vals), original[idx], side='right')
        pct = np.clip((ranks - 0.5) / len(vals), 0.01, 0.99)
        E[idx, j] = ref.s_me.mean_[j] + ref.s_me.scale_[j] * np.clip(norm.ppf(pct), -2.326, 2.326)
    both = hm & he
    mi_only = hm & ~he
    me_only = he & ~hm
    Z = np.full((len(ids), 2), np.nan)
    if both.any():
        Z[both] = ref.coords(M[both], E[both])
    if mi_only.any():
        Z[mi_only] = m.predict_method(models, 'micro', M[mi_only])
    if me_only.any():
        Z[me_only] = m.predict_method(models, 'metabo', E[me_only])
    assert np.isfinite(Z).all()
    D = squareform(pdist(Z))
    assert D.shape == (len(ids), len(ids)) and np.isfinite(D).all() and np.allclose(D, D.T) and np.allclose(np.diag(D), 0)
    out = os.path.join(os.path.dirname(HERE), f'student_res_{tag}.csv')
    pd.DataFrame(D, index=ids, columns=ids).to_csv(out, index_label='SampleID', float_format='%.7f')
    pd.DataFrame({'PC1': Z[:, 0], 'PC2': Z[:, 1], 'mode': np.select([both, mi_only, me_only], ['complete_PCA', 'micro_only_ensemble', 'metabo_only_ensemble'], default='unknown')}, index=ids).to_csv(os.path.join(os.path.dirname(HERE), f'{tag}_coordinate_audit.csv'))
    print(tag, 'total', len(ids), 'complete', both.sum(), 'mi_only', mi_only.sum(), 'me_only', me_only.sum(), 'micro_overlap', len(set(mi_cols) & set(mi.columns)), 'metabo_matches', len(matches), 'distance_range', D.max(), out)
for v, tag in [(4, 'franzosa'), (5, 'yachida')]:
    run(v, tag)
