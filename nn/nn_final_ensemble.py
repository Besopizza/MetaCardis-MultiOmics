import os
for _v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), 'data')
METHOD = 'ensemble'
OUT = os.path.join(HERE, 'final_outputs', 'best_overall_nn')
SPLIT_SEED = 42
SEEDS = [0, 1, 2]
N_PERM = 9999
BASE = dict(hidden=[128, 64], act='relu', norm='batch', dropout=0.15, lr=0.001, opt='adamw', wd=0.0001, bs=64, patience=40, max_epochs=500, loss='mse', huber_delta=1.0, input='std', es='loss', offset=False)
DIRECT = {'micro': dict(BASE, hidden=[512, 256], offset=True), 'metabo': dict(BASE, hidden=[256], act='gelu', dropout=0.5, lr=0.003, es='dist', offset=True)}
IMPUTE = {'micro': dict(BASE, wd=0.1, es='dist'), 'metabo': dict(BASE, hidden=[512, 256], norm='layer', loss='huber', input='log')}

def load_data():
    mi = pd.read_csv(os.path.join(ROOT, 'microbiome.csv'), index_col=0).astype(float)
    me = pd.read_csv(os.path.join(ROOT, 'metabolome.csv'), index_col=0).astype(float)
    meta = pd.read_csv(os.path.join(ROOT, 'metadata.csv'), index_col=0)
    me = me.loc[mi.index]
    meta = meta.loc[mi.index]
    has_mi = mi.notna().any(axis=1)
    has_me = me.notna().any(axis=1)
    complete = sorted(mi.index[has_mi & has_me])
    return (mi, me, meta, complete, has_mi, has_me)

def make_split(meta, complete):
    dev, test = train_test_split(complete, test_size=0.2, random_state=SPLIT_SEED, stratify=meta.loc[complete, 'CENTER_C'].values)
    return (sorted(dev), sorted(test))

class Reference:

    def __init__(self, mi_tr, me_tr):
        self.s_mi = StandardScaler().fit(mi_tr)
        self.s_me = StandardScaler().fit(me_tr)
        self.pca = PCA(n_components=2, svd_solver='full').fit(np.hstack([self.s_mi.transform(mi_tr), self.s_me.transform(me_tr)]))
        self.n_mi = mi_tr.shape[1]
        self.n_me = me_tr.shape[1]

    def std(self, modality, x):
        return self.s_mi.transform(x) if modality == 'micro' else self.s_me.transform(x)

    def from_std(self, xmi, xme):
        return self.pca.transform(np.hstack([xmi, xme]))

    def coords(self, mi_raw, me_raw):
        return self.from_std(self.s_mi.transform(mi_raw), self.s_me.transform(me_raw))

    def compose(self, obs, obs_raw, hidden_std):
        o = self.std(obs, obs_raw)
        return self.from_std(o, hidden_std) if obs == 'micro' else self.from_std(hidden_std, o)

    def observed_projection(self, obs, obs_raw):
        n = obs_raw.shape[0]
        return self.compose(obs, obs_raw, np.zeros((n, self.n_me if obs == 'micro' else self.n_mi)))

class InputTransform:

    def __init__(self, kind):
        self.kind = kind

    def _pre(self, x):
        x = np.asarray(x, dtype=float)
        return np.log10(np.clip(x, 0.001, None)) if self.kind == 'log' else x

    def fit(self, x):
        self.sc = StandardScaler().fit(self._pre(x))
        return self

    def transform(self, x):
        return self.sc.transform(self._pre(x))
ACTS = {'relu': nn.ReLU, 'gelu': nn.GELU}

class MLP(nn.Module):

    def __init__(self, d_in, d_out, hidden, act, norm, dropout):
        super().__init__()
        layers, d = ([], d_in)
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if norm == 'batch':
                layers.append(nn.BatchNorm1d(h))
            elif norm == 'layer':
                layers.append(nn.LayerNorm(h))
            layers.append(ACTS[act]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def upper(D):
    return D[np.triu_indices_from(D, k=1)]

def pdist_mat(Z):
    d = Z[:, None, :] - Z[None, :, :]
    return np.sqrt((d ** 2).sum(-1))

def dist_r(Zt, Zp):
    return float(pearsonr(upper(pdist_mat(Zt)), upper(pdist_mat(Zp)))[0])

def mantel(Zt, Zp, n_perm, seed):
    Dt, Dp = (pdist_mat(Zt), pdist_mat(Zp))
    iu = np.triu_indices_from(Dt, k=1)
    a = Dt[iu]
    a = (a - a.mean()) / a.std()
    b = Dp[iu]
    r_obs = float(np.mean(a * (b - b.mean()) / b.std()))
    rng = np.random.default_rng(seed)
    n = Dt.shape[0]
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        bp = Dp[p][:, p][iu]
        if np.mean(a * (bp - bp.mean()) / bp.std()) >= r_obs:
            cnt += 1
    return (r_obs, (cnt + 1) / (n_perm + 1))

def train(cfg, Xtr, Ytr, Xes, Yes, seed, es_score_fn):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xtr.shape[1], Ytr.shape[1], cfg['hidden'], cfg['act'], cfg['norm'], cfg['dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    base = nn.HuberLoss(delta=cfg['huber_delta']) if cfg['loss'] == 'huber' else nn.MSELoss()
    Xt, Yt = (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32))
    Xe, Ye = (torch.tensor(Xes, dtype=torch.float32), torch.tensor(Yes, dtype=torch.float32))
    g = torch.Generator().manual_seed(seed)
    n = len(Xt)
    best, best_state, best_ep, bad, ep = (-np.inf, None, 0, 0, 0)
    for ep in range(1, cfg['max_epochs'] + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg['bs']):
            idx = perm[i:i + cfg['bs']]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = base(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pe = model(Xe)
            score = es_score_fn(pe.numpy()) if cfg['es'] == 'dist' else -float(((pe - Ye) ** 2).mean())
        if score > best + 1e-06:
            best, best_ep, bad = (score, ep, 0)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    return (model, best_ep, ep)

def run_model(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

class Fitted:

    def __init__(self, kind, obs, cfg, ref, tf, model, best_ep, stop_ep, runtime):
        self.kind, self.obs, self.cfg, self.ref, self.tf, self.model = (kind, obs, cfg, ref, tf, model)
        self.best_ep, self.stop_ep, self.runtime = (best_ep, stop_ep, runtime)

    def predict(self, obs_raw):
        out = run_model(self.model, self.tf.transform(obs_raw))
        if self.kind == 'direct':
            return out + (self.ref.observed_projection(self.obs, obs_raw) if self.cfg['offset'] else 0.0)
        return self.ref.compose(self.obs, obs_raw, out)

    def predict_hidden_features(self, obs_raw):
        return run_model(self.model, self.tf.transform(obs_raw)) if self.kind == 'impute' else None

def fit(kind, obs, cfg, mi, me, tr_ids, seed):
    ref = Reference(mi.loc[tr_ids].values, me.loc[tr_ids].values)
    in_tr, es_ids = train_test_split(tr_ids, test_size=0.15, random_state=seed)
    src = mi if obs == 'micro' else me
    hid = me if obs == 'micro' else mi
    hid_mod = 'metabo' if obs == 'micro' else 'micro'
    tf = InputTransform(cfg['input']).fit(src.loc[in_tr].values)
    Xin, Xes = (tf.transform(src.loc[in_tr].values), tf.transform(src.loc[es_ids].values))
    Zes = ref.coords(mi.loc[es_ids].values, me.loc[es_ids].values)
    es_raw = src.loc[es_ids].values
    if kind == 'direct':
        Ytr = ref.coords(mi.loc[in_tr].values, me.loc[in_tr].values)
        Yes = Zes
        off_es = 0.0
        if cfg['offset']:
            off_es = ref.observed_projection(obs, es_raw)
            Ytr = Ytr - ref.observed_projection(obs, src.loc[in_tr].values)
            Yes = Yes - off_es
        es_fn = lambda p: dist_r(Zes, p + off_es)
    else:
        Ytr = ref.std(hid_mod, hid.loc[in_tr].values)
        Yes = ref.std(hid_mod, hid.loc[es_ids].values)
        es_fn = lambda p: dist_r(Zes, ref.compose(obs, es_raw, p))
    t0 = time.time()
    model, best_ep, stop_ep = train(cfg, Xin, Ytr, Xes, Yes, seed, es_fn)
    return Fitted(kind, obs, cfg, ref, tf, model, best_ep, stop_ep, time.time() - t0)

def fit_method(mi, me, tr_ids, seed):
    models = {}
    for obs in ['micro', 'metabo']:
        if METHOD in ('direct', 'ensemble'):
            models['direct', obs] = fit('direct', obs, DIRECT[obs], mi, me, tr_ids, seed)
        if METHOD in ('impute', 'ensemble'):
            models['impute', obs] = fit('impute', obs, IMPUTE[obs], mi, me, tr_ids, seed)
    return models

def predict_method(models, obs, obs_raw):
    preds = [m.predict(obs_raw) for (k, o), m in models.items() if o == obs]
    return np.mean(preds, axis=0)

def scenario_labels(n, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_c = int(round(0.6 * n))
    n_m = (n - n_c) // 2
    mixed = np.empty(n, dtype=object)
    mixed[perm[:n_c]], mixed[perm[n_c:n_c + n_m]], mixed[perm[n_c + n_m:]] = ('complete', 'micro', 'metabo')
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    half = np.empty(n, dtype=object)
    half[perm[:n // 2]], half[perm[n // 2:]] = ('micro', 'metabo')
    return (half, mixed)

def evaluate_internal_test(mi, me, dev, test):
    rows = []
    for seed in SEEDS:
        models = fit_method(mi, me, dev, seed)
        ref = next(iter(models.values())).ref
        Zt = ref.coords(mi.loc[test].values, me.loc[test].values)
        Zmi = predict_method(models, 'micro', mi.loc[test].values)
        Zme = predict_method(models, 'metabo', me.loc[test].values)
        half, mixed = scenario_labels(len(test))
        Zh = np.where((half == 'micro')[:, None], Zmi, Zme)
        Zm = np.where((mixed == 'complete')[:, None], Zt, np.where((mixed == 'micro')[:, None], Zmi, Zme))
        row = dict(method=METHOD, seed=seed, n_train=len(dev), n_test=len(test))
        for name, Zp in [('S_micro', Zmi), ('S_metabo', Zme), ('S_half', Zh), ('S_mixed', Zm)]:
            a, b = (upper(pdist_mat(Zt)), upper(pdist_mat(Zp)))
            row[f'{name}_dist_pearson_r'] = float(pearsonr(a, b)[0])
            row[f'{name}_dist_spearman_rho'] = float(spearmanr(a, b)[0])
            row[f'{name}_mantel_r'], row[f'{name}_mantel_p'] = mantel(Zt, Zp, N_PERM, seed)
            row[f'{name}_coord_mse'] = float(np.mean((Zt - Zp) ** 2))
        row['primary_masked_mean_r'] = (row['S_micro_dist_pearson_r'] + row['S_metabo_dist_pearson_r']) / 2
        for (k, o), m in models.items():
            row[f'{k}_{o}_best_epoch'], row[f'{k}_{o}_stop_epoch'], row[f'{k}_{o}_train_seconds'] = (m.best_ep, m.stop_ep, m.runtime)
            if k == 'impute':
                hid_raw = me.loc[test].values if o == 'micro' else mi.loc[test].values
                true_std = ref.std('metabo' if o == 'micro' else 'micro', hid_raw)
                row[f'impute_{o}_feature_mse'] = float(np.mean((m.predict_hidden_features(mi.loc[test].values if o == 'micro' else me.loc[test].values) - true_std) ** 2))
        rows.append(row)
        print(f"seed {seed}: S_micro r={row['S_micro_dist_pearson_r']:.4f}  S_metabo r={row['S_metabo_dist_pearson_r']:.4f}  S_half r={row['S_half_dist_pearson_r']:.4f}  S_mixed r={row['S_mixed_dist_pearson_r']:.4f}", flush=True)
    return pd.DataFrame(rows)

def deploy(mi, me, complete, has_mi, has_me, seed=0):
    models = fit_method(mi, me, complete, seed)
    ref = next(iter(models.values())).ref
    ids = list(mi.index)
    coords = pd.DataFrame(index=ids, columns=['PC1', 'PC2'], dtype=float)
    source = pd.Series(index=ids, dtype=object)
    both = [i for i in ids if has_mi[i] and has_me[i]]
    mo = [i for i in ids if has_mi[i] and (not has_me[i])]
    eo = [i for i in ids if has_me[i] and (not has_mi[i])]
    coords.loc[both] = ref.coords(mi.loc[both].values, me.loc[both].values)
    source.loc[both] = 'observed_microbiome_and_metabolome'
    if mo:
        coords.loc[mo] = predict_method(models, 'micro', mi.loc[mo].values)
        source.loc[mo] = 'predicted_from_microbiome_only'
    if eo:
        coords.loc[eo] = predict_method(models, 'metabo', me.loc[eo].values)
        source.loc[eo] = 'predicted_from_metabolome_only'
    coords['source'] = source
    coords = coords.dropna(subset=['PC1', 'PC2'])
    coords.to_csv(os.path.join(OUT, 'deployment_patient_coordinates.csv'), index_label='SampleID')
    D = pdist_mat(coords[['PC1', 'PC2']].values.astype(float))
    pd.DataFrame(D, index=coords.index, columns=coords.index).to_csv(os.path.join(OUT, 'deployment_distance_matrix.csv'), index_label='SampleID', float_format='%.5f')
    with open(os.path.join(OUT, 'deployment_models.pkl'), 'wb') as f:
        pickle.dump({'method': METHOD, 'models': models, 'microbiome_features': list(mi.columns), 'metabolome_features': list(me.columns)}, f)
    return coords

class ModelUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == '__main__' and name in globals():
            return globals()[name]
        return super().find_class(module, name)

def predict_external(model_path, microbiome_csv=None, metabolome_csv=None):
    with open(model_path, 'rb') as f:
        bundle = ModelUnpickler(f).load()
    models, mi_cols, me_cols = (bundle['models'], bundle['microbiome_features'], bundle['metabolome_features'])
    ref = next(iter(models.values())).ref
    mi = pd.read_csv(microbiome_csv, index_col=0).astype(float) if microbiome_csv else pd.DataFrame(columns=mi_cols, dtype=float)
    me = pd.read_csv(metabolome_csv, index_col=0).astype(float) if metabolome_csv else pd.DataFrame(columns=me_cols, dtype=float)
    ids = sorted(set(mi.index) | set(me.index))
    mi = mi.reindex(index=ids, columns=mi_cols)
    me = me.reindex(index=ids, columns=me_cols)
    hm, he = (mi.notna().all(axis=1).values, me.notna().all(axis=1).values)
    Z = np.full((len(ids), 2), np.nan)
    if (hm & he).any():
        Z[hm & he] = ref.coords(mi.values[hm & he], me.values[hm & he])
    if (hm & ~he).any():
        Z[hm & ~he] = predict_method(models, 'micro', mi.values[hm & ~he])
    if (he & ~hm).any():
        Z[he & ~hm] = predict_method(models, 'metabo', me.values[he & ~hm])
    coords = pd.DataFrame(Z, index=ids, columns=['PC1', 'PC2']).dropna()
    return (coords, pd.DataFrame(pdist_mat(coords.values), index=coords.index, columns=coords.index))
if __name__ == '__main__' and len(sys.argv) == 5 and (sys.argv[1] == 'predict'):
    os.makedirs(sys.argv[4], exist_ok=True)
    coords, dist = predict_external(os.path.join(OUT, 'deployment_models.pkl'), None if sys.argv[2] == '-' else sys.argv[2], None if sys.argv[3] == '-' else sys.argv[3])
    coords.to_csv(os.path.join(sys.argv[4], 'external_patient_coordinates.csv'), index_label='SampleID')
    dist.to_csv(os.path.join(sys.argv[4], 'external_distance_matrix.csv'), index_label='SampleID', float_format='%.5f')
    print(coords.shape)
elif __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    mi, me, meta, complete, has_mi, has_me = load_data()
    dev, test = make_split(meta, complete)
    t0 = time.time()
    res = evaluate_internal_test(mi, me, dev, test)
    res.to_csv(os.path.join(OUT, 'internal_test_results_per_seed.csv'), index=False)
    summary = res.drop(columns=['method']).agg(['mean', 'std']).T
    summary.to_csv(os.path.join(OUT, 'internal_test_results_summary.csv'))
    print(summary.loc[[c for c in summary.index if c.endswith('pearson_r') or c.startswith('primary')]].round(4).to_string())
    coords = deploy(mi, me, complete, has_mi, has_me)
    print(coords['source'].value_counts().to_string())
    print(f'total runtime {time.time() - t0:.1f}s')
