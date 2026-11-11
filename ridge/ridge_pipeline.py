import pandas as pd
import numpy as np
metadata = pd.read_csv('metadata.csv', index_col=0)
microbiome = pd.read_csv('microbiome.csv', index_col=0)
metabolome = pd.read_csv('metabolome.csv', index_col=0)
print(f'Metadata shape: {metadata.shape} (Patients, Clinical Features)')
print(f'Microbiome shape: {microbiome.shape} (Patients, Bacterial Species)')
print(f'Metabolome shape: {metabolome.shape} (Patients, Metabolites)')
print('\nFirst 5 rows of metadata:')
print(metadata.head())
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
metadata = pd.read_csv('metadata.csv', index_col=0)
microbiome = pd.read_csv('microbiome.csv', index_col=0)
metabolome = pd.read_csv('metabolome.csv', index_col=0)
micro_features = microbiome.columns
metab_features = metabolome.columns
complete_idx = microbiome.dropna(how='all').index.intersection(metabolome.dropna(how='all').index)
print(f'Total samples: {len(metadata)}')
print(f'Complete samples with both omics: {len(complete_idx)}')
X_micro = microbiome.loc[complete_idx].values
Y_metab = metabolome.loc[complete_idx].values
X_tr_micro, X_te_micro, Y_tr_metab, Y_te_metab = train_test_split(X_micro, Y_metab, test_size=0.2, random_state=42)
scaler_micro = StandardScaler().fit(X_tr_micro)
scaler_metab = StandardScaler().fit(Y_tr_metab)
X_tr_micro_sc = scaler_micro.transform(X_tr_micro)
X_te_micro_sc = scaler_micro.transform(X_te_micro)
Y_tr_metab_sc = scaler_metab.transform(Y_tr_metab)
Y_te_metab_sc = scaler_metab.transform(Y_te_metab)
print('\n--- Training Linear Regression (Ridge) ---')
start_time = time.time()
ridge_m2m = RidgeCV(alphas=np.logspace(-2, 4, 20))
ridge_m2m.fit(X_tr_micro_sc, Y_tr_metab_sc)
pred_te_metab_ridge = ridge_m2m.predict(X_te_micro_sc)
ridge_m2b = RidgeCV(alphas=np.logspace(-2, 4, 20))
ridge_m2b.fit(Y_tr_metab_sc, X_tr_micro_sc)
pred_te_micro_ridge = ridge_m2b.predict(Y_te_metab_sc)
ridge_runtime = time.time() - start_time
mse_m2m_ridge = mean_squared_error(Y_te_metab_sc, pred_te_metab_ridge)
r2_m2m_ridge = r2_score(Y_te_metab_sc, pred_te_metab_ridge)
print(f'Ridge Micro -> Metab | Test MSE: {mse_m2m_ridge:.4f} | R2: {r2_m2m_ridge:.4f}')
print(f'Ridge Total Training & Inference Time: {ridge_runtime:.2f} seconds')
print('\n--- Training Random Forest Regressor ---')
start_time = time.time()
rf_m2m = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_m2m.fit(X_tr_micro_sc, Y_tr_metab_sc)
pred_te_metab_rf = rf_m2m.predict(X_te_micro_sc)
rf_m2b = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_m2b.fit(Y_tr_metab_sc, X_tr_micro_sc)
pred_te_micro_rf = rf_m2b.predict(Y_te_metab_sc)
rf_runtime = time.time() - start_time
mse_m2m_rf = mean_squared_error(Y_te_metab_sc, pred_te_metab_rf)
r2_m2m_rf = r2_score(Y_te_metab_sc, pred_te_metab_rf)
print(f'RF Micro -> Metab | Test MSE: {mse_m2m_rf:.4f} | R2: {r2_m2m_rf:.4f}')
print(f'RF Total Training & Inference Time: {rf_runtime:.2f} seconds')
train_combined_sc = np.hstack([X_tr_micro_sc, Y_tr_metab_sc])
pca_pipeline = PCA(n_components=2, random_state=42)
gt_pca_train = pca_pipeline.fit_transform(train_combined_sc)
test_true_combined_sc = np.hstack([X_te_micro_sc, Y_te_metab_sc])
test_true_pca = pca_pipeline.transform(test_true_combined_sc)
dist_true_test = squareform(pdist(test_true_pca, metric='euclidean'))
gt_pca_tr = pca_pipeline.transform(train_combined_sc)
ridge_pca_m = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_tr_micro_sc, gt_pca_tr)
ridge_pca_b = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(Y_tr_metab_sc, gt_pca_tr)
rf_pca_m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr_micro_sc, gt_pca_tr)
rf_pca_b = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(Y_tr_metab_sc, gt_pca_tr)
print('Task B models trained successfully!')

def evaluate_task_a_true_leak_free(model_m2m, model_m2b, model_name):
    n_test = len(X_te_micro_sc)
    idx_micro_only = np.arange(0, n_test // 2)
    idx_metab_only = np.arange(n_test // 2, n_test)
    pred_metab_g1 = model_m2m.predict(X_te_micro_sc[idx_micro_only])
    g1_combined = np.hstack([X_te_micro_sc[idx_micro_only], pred_metab_g1])
    pred_micro_g2 = model_m2b.predict(Y_te_metab_sc[idx_metab_only])
    g2_combined = np.hstack([pred_micro_g2, Y_te_metab_sc[idx_metab_only]])
    test_imputed_combined_sc = np.vstack([g1_combined, g2_combined])
    test_imputed_pca = pca_pipeline.transform(test_imputed_combined_sc)
    dist_imputed_test = squareform(pdist(test_imputed_pca, metric='euclidean'))
    triu_idx = np.triu_indices_from(dist_true_test, k=1)
    r_score, _ = spearmanr(dist_true_test[triu_idx], dist_imputed_test[triu_idx])
    print(f'\n=== Task A ({model_name} True Leak-Free) ===')
    print(f'Test Set Mantel Correlation (r): {r_score:.4f}')
    return r_score
r_ridge_task_a = evaluate_task_a_true_leak_free(ridge_m2m, ridge_m2b, 'Ridge')
r_rf_task_a = evaluate_task_a_true_leak_free(rf_m2m, rf_m2b, 'Random Forest')

def evaluate_task_b_true_leak_free(model_m, model_b, model_name):
    n_test = len(X_te_micro_sc)
    idx_micro_only = np.arange(0, n_test // 2)
    idx_metab_only = np.arange(n_test // 2, n_test)
    pred_pca_g1 = model_m.predict(X_te_micro_sc[idx_micro_only])
    pred_pca_g2 = model_b.predict(Y_te_metab_sc[idx_metab_only])
    pred_pca_combined = np.vstack([pred_pca_g1, pred_pca_g2])
    dist_pred_test = squareform(pdist(pred_pca_combined, metric='euclidean'))
    triu_idx = np.triu_indices_from(dist_true_test, k=1)
    r_score, _ = spearmanr(dist_true_test[triu_idx], dist_pred_test[triu_idx])
    print(f'\n=== Task B ({model_name} True Leak-Free) ===')
    print(f'Test Set Mantel Correlation (r): {r_score:.4f}')
    return r_score
r_ridge_task_b = evaluate_task_b_true_leak_free(ridge_pca_m, ridge_pca_b, 'Ridge')
r_rf_task_b = evaluate_task_b_true_leak_free(rf_pca_m, rf_pca_b, 'Random Forest')

def align_features(df_input, target_columns):
    aligned = pd.DataFrame(index=df_input.index)
    for col in target_columns:
        if col in df_input.columns:
            aligned[col] = df_input[col]
        else:
            aligned[col] = 0.0
    return aligned

def save_ridge_pca_distance_matrix(output_filename='metacardis_res_student.csv'):
    full_micro = microbiome.reindex(metadata.index).copy()
    full_metab = metabolome.reindex(metadata.index).copy()
    micro_only_pts = full_micro.dropna(how='all').index.difference(complete_idx)
    metab_only_pts = full_metab.dropna(how='all').index.difference(complete_idx)
    scaled_micro_avail = scaler_micro.transform(full_micro.loc[micro_only_pts].fillna(0).values)
    scaled_metab_avail = scaler_metab.transform(full_metab.loc[metab_only_pts].fillna(0).values)
    scaled_complete_micro = scaler_micro.transform(full_micro.loc[complete_idx].fillna(0).values)
    scaled_complete_metab = scaler_metab.transform(full_metab.loc[complete_idx].fillna(0).values)
    complete_combined_sc = np.hstack([scaled_complete_micro, scaled_complete_metab])
    pca_complete = pca_pipeline.transform(complete_combined_sc)
    pca_micro_only = ridge_pca_m.predict(scaled_micro_avail)
    pca_metab_only = ridge_pca_b.predict(scaled_metab_avail)
    df_pca_all = pd.DataFrame(index=complete_idx.tolist() + micro_only_pts.tolist() + metab_only_pts.tolist(), data=np.vstack([pca_complete, pca_micro_only, pca_metab_only])).reindex(metadata.index)
    dist_matrix = squareform(pdist(df_pca_all.values, metric='euclidean'))
    df_dist = pd.DataFrame(dist_matrix, index=metadata.index, columns=metadata.index)
    df_dist.to_csv(output_filename)
    print(f"✓ Successfully generated and saved '{output_filename}' with shape: {df_dist.shape}")
save_ridge_pca_distance_matrix('metacardis_res_student.csv')
micro_franzosa = pd.read_csv('microbiome_Franzosa.csv', index_col=0)
metab_franzosa = pd.read_csv('metabolome_Franzosa.csv', index_col=0)
meta_franzosa = pd.read_csv('metadata_Franzosa.csv', index_col=0)
micro_f_aligned = align_features(micro_franzosa, micro_features)
metab_f_aligned = align_features(metab_franzosa, metab_features)
X_f_micro_sc = scaler_micro.transform(micro_f_aligned.fillna(0).values)
Y_f_metab_sc = scaler_metab.transform(metab_f_aligned.fillna(0).values)
X_f_combined = np.hstack([X_f_micro_sc, Y_f_metab_sc])
pca_franzosa_2d = pca_pipeline.transform(X_f_combined)
dist_franzosa = squareform(pdist(pca_franzosa_2d, metric='euclidean'))
df_franzosa_res = pd.DataFrame(dist_franzosa, index=meta_franzosa.index, columns=meta_franzosa.index)
df_franzosa_res.to_csv('franzosa_res_student.csv')
print(f" Successfully generated 'franzosa_res_student.csv' ({df_franzosa_res.shape})")
micro_yachida = pd.read_csv('microbiome_Yachida.csv', index_col=0)
metab_yachida = pd.read_csv('metabolome_Yachida.csv', index_col=0)
meta_yachida = pd.read_csv('metadata_Yachida.csv', index_col=0)
micro_y_aligned = align_features(micro_yachida, micro_features)
metab_y_aligned = align_features(metab_yachida, metab_features)
X_y_micro_sc = scaler_micro.transform(micro_y_aligned.fillna(0).values)
Y_y_metab_sc = scaler_metab.transform(metab_y_aligned.fillna(0).values)
X_y_combined = np.hstack([X_y_micro_sc, Y_y_metab_sc])
pca_yachida_2d = pca_pipeline.transform(X_y_combined)
dist_yachida = squareform(pdist(pca_yachida_2d, metric='euclidean'))
df_yachida_res = pd.DataFrame(dist_yachida, index=meta_yachida.index, columns=meta_yachida.index)
df_yachida_res.to_csv('yachida_res_student.csv')
print(f" Successfully generated 'yachida_res_student.csv' ({df_yachida_res.shape})")
from sklearn.metrics import mean_squared_error

def stress_test(model_m2m, X_test_sc, Y_test_sc, missing_rates=[0.1, 0.3, 0.5, 0.7]):
    print('\n--- Missingness Stress Test (Microbiome -> Metabolome) ---')
    base_pred = model_m2m.predict(X_test_sc)
    base_mse = mean_squared_error(Y_test_sc, base_pred)
    print(f'Baseline (0% Masked): MSE = {base_mse:.4f}')
    np.random.seed(42)
    for rate in missing_rates:
        X_masked = X_test_sc.copy()
        mask = np.random.rand(*X_masked.shape) < rate
        X_masked[mask] = 0.0
        pred = model_m2m.predict(X_masked)
        mse = mean_squared_error(Y_test_sc, pred)
        pct_change = (mse - base_mse) / base_mse * 100
        print(f'Masking rate {int(rate * 100)}%: Test MSE = {mse:.4f} (+{pct_change:.1f}%)')
stress_test(rf_m2m, X_te_micro_sc, Y_te_metab_sc)
