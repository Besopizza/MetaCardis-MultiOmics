# MetaCardis — multi-omics distance estimation

The repository contains the original direct neural network, its improved direct and imputation first coordinate ensemble, frozen external inference, and the classical Ridge/Random Forest.
## Code

- `nn/nn_original_direct.py`: original Milestone 2 MLP implementation 
- `nn/nn_final_ensemble.py`: final training/evaluation/export code. Includes four learned networks direct/microbiome, direct/metabolome, feature-imputation/microbiome, feature-imputation/metabolome. Uses mean of the two predicted PCA coordinates for single-modality samples.
- `nn/nn_frozen_external_inference.py`: zero shot inference on Franzosa and Yachida with the saved MetaCardis fitted model uses feature matching and approximate metabolomics mapping does not call `fit` or train any model.
- `nn/final_outputs/best_overall_nn/deployment_models.pkl`: frozen trained neural-network bundle.
- `ridge/milestone3.ipynb`: ridge notebook.
- `ridge/ridge_pipeline.py`: executable Python riddge. Running it trains the classical models.

## Commands

Install dependencies: `python3 -m pip install -r requirements.txt`

Original baseline (trains): `python3 nn/nn_original_direct.py`

Final ensemble evaluation and deployment (trains): `python3 nn/nn_final_ensemble.py`

Frozen NN inference without training (uses saved fitted bundle): `python3 nn/nn_frozen_external_inference.py`

The final ensemble script also has a `predict` mode for external features but its built-in version expects a fully matching input feature panel the dedicated frozen-inference script instead applies the provisional feature alignment strategy used in the submitted external outputs.
