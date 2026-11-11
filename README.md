# MetaCardis — multi-omics distance estimation

The repository contains the original direct neural network, its improved direct and imputation-first coordinate ensemble, frozen external inference, and the classical Ridge/Random Forest analysis. The scripts contain no inline comments or docstrings. This README describes their purpose separately.

## Code

- `nn/nn_original_direct.py`: original Milestone 2 MLP implementation; retained as a historical baseline.
- `nn/nn_final_ensemble.py`: final training/evaluation/export code. Includes four learned networks: direct/microbiome, direct/metabolome, feature-imputation/microbiome, feature-imputation/metabolome. Uses mean of the two predicted PCA coordinates for single-modality samples.
- `nn/nn_frozen_external_inference.py`: zero-shot inference on Franzosa and Yachida with the saved MetaCardis fitted-model bundle. Uses feature matching and approximate metabolomics mapping. Does not call `fit` or train any model. Its external generalization is not guaranteed.
- `nn/final_outputs/best_overall_nn/deployment_models.pkl`: frozen trained neural-network bundle. Load only if you trust the source; pickle can run arbitrary code.
- `ridge/milestone3.ipynb`: partner's notebook, preserved as supplied.
- `ridge/ridge_pipeline.py`: executable Python extracted from that notebook; removes the notebook-only `pip install` cell and comments/docstrings, but preserves its computational cells. Running it trains the classical models.

## Data layout

Do **not** upload raw patient-level data to a public repository without permission. Keep the three MetaCardis input files locally in `data/` named `microbiome.csv`, `metabolome.csv`, and `metadata.csv`. For frozen external inference, add the six external dataset files as `microbiome(4).csv`, `metabolome(4).csv`, and corresponding `(5)` files. In the project conversation, `(4)` is Franzosa and `(5)` is Yachida.

## Commands

Install dependencies: `python3 -m pip install -r requirements.txt`

Original baseline (trains): `python3 nn/nn_original_direct.py`

Final ensemble evaluation and deployment (trains): `python3 nn/nn_final_ensemble.py`

Frozen NN inference without training (uses saved fitted bundle): `python3 nn/nn_frozen_external_inference.py`

The final ensemble script also has a `predict` mode for external features, but its built-in version expects a fully matching input feature panel. The dedicated frozen-inference script instead applies the provisional feature-alignment strategy used in the submitted external outputs.

## Scientific notes

The MetaCardis internal held-out evaluation used training-only fitted StandardScaler and PCA(n_components=2), then whole-modality masking. The ensemble combines **coordinates**, not distance matrices; complete patients are projected using observed combined modalities. The professors' evaluation did not show consistent superiority over mean imputation or Random Forest. The external feature mismatch is a material limitation.

## Submission

The course instructions ask for a 4–5 page scientific PDF report (Introduction, Methods, Results, Discussion and Bibliography), GitHub code, and CSV predictions. The personal project reflection is a different format and should not be substituted for the scientific report without the instructors' permission.
