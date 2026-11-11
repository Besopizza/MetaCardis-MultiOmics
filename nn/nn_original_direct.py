import os
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
MICROBIOME_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'microbiome.csv')
METABOLOME_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'metabolome.csv')
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'metadata.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'initial_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DISTANCE_MATRIX_OUT = os.path.join(OUTPUT_DIR, 'nn_pca2_distance_matrix.csv')
COORDINATES_OUT = os.path.join(OUTPUT_DIR, 'nn_pca2_patient_coordinates.csv')
EVAL_OUT = os.path.join(OUTPUT_DIR, 'nn_internal_holdout_results.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

def load_table(path):
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'SampleID'})
    elif df.columns[0] != 'SampleID':
        df = df.rename(columns={df.columns[0]: 'SampleID'})
    return df
microbiome = load_table(MICROBIOME_PATH)
metabolome = load_table(METABOLOME_PATH)
metadata = load_table(METADATA_PATH)
micro_features = [c for c in microbiome.columns if c != 'SampleID']
metabo_features = [c for c in metabolome.columns if c != 'SampleID']
micro_df = microbiome.set_index('SampleID')[micro_features].astype(float)
metabo_df = metabolome.set_index('SampleID')[metabo_features].astype(float)
all_sample_ids = sorted(set(micro_df.index).union(set(metabo_df.index)))
micro_available = micro_df.notna().any(axis=1).reindex(all_sample_ids).fillna(False)
metabo_available = metabo_df.notna().any(axis=1).reindex(all_sample_ids).fillna(False)
availability = pd.DataFrame({'has_microbiome': micro_available.astype(bool), 'has_metabolome': metabo_available.astype(bool)}, index=all_sample_ids)
print('\nModality availability:')
print(availability.value_counts())
complete_ids = availability.query('has_microbiome and has_metabolome').index.tolist()
micro_only_ids = availability.query('has_microbiome and not has_metabolome').index.tolist()
metabo_only_ids = availability.query('has_metabolome and not has_microbiome').index.tolist()
print('\nComplete samples:', len(complete_ids))
print('Microbiome-only samples:', len(micro_only_ids))
print('Metabolome-only samples:', len(metabo_only_ids))
print('Neither missing:', len(availability.query('not has_microbiome and not has_metabolome')))

class SimpleMLP(nn.Module):

    def __init__(self, input_dim, output_dim=2, hidden1=128, hidden2=64, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden2, output_dim))

    def forward(self, x):
        return self.net(x)

def train_model(X_train, y_train, X_val, y_val, input_dim, epochs=500, batch_size=64, lr=0.001, weight_decay=0.0001, patience=40):
    model = SimpleMLP(input_dim=input_dim, output_dim=y_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    best_val = np.inf
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            n_train += len(xb)
        train_loss = train_loss_sum / n_train
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        if val_loss < best_val - 1e-05:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    return (model, pd.DataFrame(history))

def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(X_t).cpu().numpy()

def distance_correlation(Z_true, Z_pred):
    D_true = pairwise_distances(Z_true)
    D_pred = pairwise_distances(Z_pred)
    idx = np.triu_indices_from(D_true, k=1)
    corr = pearsonr(D_true[idx], D_pred[idx])[0]
    return corr

def coord_metrics(Z_true, Z_pred, name):
    return {'model': name, 'coord_rmse': float(np.sqrt(mean_squared_error(Z_true, Z_pred))), 'coord_r2': float(r2_score(Z_true, Z_pred)), 'distance_correlation': float(distance_correlation(Z_true, Z_pred))}
train_ids, test_ids = train_test_split(complete_ids, test_size=0.2, random_state=SEED)
micro_scaler = StandardScaler()
metabo_scaler = StandardScaler()
X_micro_train = micro_scaler.fit_transform(micro_df.loc[train_ids].values)
X_micro_test = micro_scaler.transform(micro_df.loc[test_ids].values)
X_metabo_train = metabo_scaler.fit_transform(metabo_df.loc[train_ids].values)
X_metabo_test = metabo_scaler.transform(metabo_df.loc[test_ids].values)
combined_train = np.hstack([X_micro_train, X_metabo_train])
combined_test = np.hstack([X_micro_test, X_metabo_test])
pca = PCA(n_components=2, random_state=SEED)
Z_train = pca.fit_transform(combined_train)
Z_test_true = pca.transform(combined_test)
print('\nInternal PCA explained variance ratio:', pca.explained_variance_ratio_)
train_idx, val_idx = train_test_split(np.arange(len(train_ids)), test_size=0.15, random_state=SEED)
X_micro_tr, X_micro_val = (X_micro_train[train_idx], X_micro_train[val_idx])
X_metabo_tr, X_metabo_val = (X_metabo_train[train_idx], X_metabo_train[val_idx])
Z_tr, Z_val = (Z_train[train_idx], Z_train[val_idx])
t0 = time.time()
micro_to_z_model, micro_hist = train_model(X_micro_tr, Z_tr, X_micro_val, Z_val, input_dim=X_micro_train.shape[1])
metabo_to_z_model, metabo_hist = train_model(X_metabo_tr, Z_tr, X_metabo_val, Z_val, input_dim=X_metabo_train.shape[1])
print(f'\nTraining time for internal models: {time.time() - t0:.2f} seconds')
print('Microbiome → PCA2 epochs:', len(micro_hist))
print('Metabolome → PCA2 epochs:', len(metabo_hist))
Z_pred_from_micro = predict(micro_to_z_model, X_micro_test)
Z_pred_from_metabo = predict(metabo_to_z_model, X_metabo_test)
Z_pred_average = (Z_pred_from_micro + Z_pred_from_metabo) / 2.0
eval_results = pd.DataFrame([coord_metrics(Z_test_true, Z_pred_from_micro, 'microbiome_only_to_PCA2'), coord_metrics(Z_test_true, Z_pred_from_metabo, 'metabolome_only_to_PCA2'), coord_metrics(Z_test_true, Z_pred_average, 'average_of_both_predictions')])
print('\nInternal holdout results:')
print(eval_results)
eval_results.to_csv(EVAL_OUT, index=False)
print('\nSaved internal evaluation to:', EVAL_OUT)
final_micro_scaler = StandardScaler()
final_metabo_scaler = StandardScaler()
X_micro_complete = final_micro_scaler.fit_transform(micro_df.loc[complete_ids].values)
X_metabo_complete = final_metabo_scaler.fit_transform(metabo_df.loc[complete_ids].values)
final_combined_complete = np.hstack([X_micro_complete, X_metabo_complete])
final_pca = PCA(n_components=2, random_state=SEED)
Z_complete = final_pca.fit_transform(final_combined_complete)
print('\nFinal PCA explained variance ratio:', final_pca.explained_variance_ratio_)
final_train_idx, final_val_idx = train_test_split(np.arange(len(complete_ids)), test_size=0.15, random_state=SEED)
final_micro_model, final_micro_hist = train_model(X_micro_complete[final_train_idx], Z_complete[final_train_idx], X_micro_complete[final_val_idx], Z_complete[final_val_idx], input_dim=X_micro_complete.shape[1])
final_metabo_model, final_metabo_hist = train_model(X_metabo_complete[final_train_idx], Z_complete[final_train_idx], X_metabo_complete[final_val_idx], Z_complete[final_val_idx], input_dim=X_metabo_complete.shape[1])
print('\nFinal models trained.')
print('Final microbiome model epochs:', len(final_micro_hist))
print('Final metabolome model epochs:', len(final_metabo_hist))
coords = pd.DataFrame(index=all_sample_ids, columns=['PC1_pred', 'PC2_pred', 'source'], dtype=object)
both_ids = complete_ids
if len(both_ids) > 0:
    micro_scaled_both = final_micro_scaler.transform(micro_df.loc[both_ids].values)
    metabo_scaled_both = final_metabo_scaler.transform(metabo_df.loc[both_ids].values)
    combined_scaled_both = np.hstack([micro_scaled_both, metabo_scaled_both])
    z_both = final_pca.transform(combined_scaled_both)
    coords.loc[both_ids, ['PC1_pred', 'PC2_pred']] = z_both
    coords.loc[both_ids, 'source'] = 'real_microbiome_and_real_metabolome'
if len(micro_only_ids) > 0:
    micro_scaled_only = final_micro_scaler.transform(micro_df.loc[micro_only_ids].values)
    z_micro_only = predict(final_micro_model, micro_scaled_only)
    coords.loc[micro_only_ids, ['PC1_pred', 'PC2_pred']] = z_micro_only
    coords.loc[micro_only_ids, 'source'] = 'predicted_from_microbiome_only'
if len(metabo_only_ids) > 0:
    metabo_scaled_only = final_metabo_scaler.transform(metabo_df.loc[metabo_only_ids].values)
    z_metabo_only = predict(final_metabo_model, metabo_scaled_only)
    coords.loc[metabo_only_ids, ['PC1_pred', 'PC2_pred']] = z_metabo_only
    coords.loc[metabo_only_ids, 'source'] = 'predicted_from_metabolome_only'
if coords[['PC1_pred', 'PC2_pred']].isna().any().any():
    missing = coords[coords[['PC1_pred', 'PC2_pred']].isna().any(axis=1)]
    raise ValueError(f'Some samples did not receive coordinates. Count: {len(missing)}')
coords[['PC1_pred', 'PC2_pred']] = coords[['PC1_pred', 'PC2_pred']].astype(float)
coords.to_csv(COORDINATES_OUT, index_label='SampleID')
print('\nSaved patient coordinates to:', COORDINATES_OUT)
print(coords['source'].value_counts())
Z_all = coords[['PC1_pred', 'PC2_pred']].values.astype(float)
distance_matrix = pairwise_distances(Z_all, metric='euclidean')
distance_df = pd.DataFrame(distance_matrix, index=coords.index, columns=coords.index)
distance_df.to_csv(DISTANCE_MATRIX_OUT, index_label='SampleID')
print('\nSaved distance matrix to:', DISTANCE_MATRIX_OUT)
print('Distance matrix shape:', distance_df.shape)
print('\nDone.')
