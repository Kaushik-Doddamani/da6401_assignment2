# DA6401_Intro_to_DeepLearning_Assignment_2

## 📁 Project Overview

This repository contains a complete implementation for the iNaturalist 12K classification assignment for course: DA6401 Introduciton to Deep Learning, IIT Madras. It covers:

- **Part A**: Training a custom CNN from scratch, including hyperparameter sweeps.
- **Part B**: Fine‑tuning a pre‑trained ResNet‑50 with a progressive freeze/unfreeze strategy.

The codebase is organized into Python modules, YAML configuration files, and Jupyter notebooks for interactive exploration.

Please find Wandb Report [here](https://wandb.ai/da24s020-indian-institute-of-technology-madras/DA6401_Intro_to_DeepLearning_Assignment_2/reports/DA6401-Assignment-2--VmlldzoxMjMxODA4Mw?accessToken=n3jv7pu7k48k0q6503sooe6jfvn4sjjlepamilnanl3m97fckdx39mudcmb1xpqv)

---

## 📂 Directory Structure

```
DA6401_Intro_to_DeepLearning_Assignment_2/
├─ config/
│  ├─ configs.yaml         # Static configs for each solution
│  └─ sweep_config.yaml    # W&B sweep hyperparameter grid
│
├─ data/
│  └─ data_loader.py       # Lightning and custom DataLoader utilities
│
├─ models/
│  ├─ implementation.py    # MyCNN and MyCNNExtended architectures
│  └─ wrapper.py           # PyTorch Lightning wrapper (LitInatModel)
│
├─ notebooks/              # Original .ipynb files can be found here
│  ├─ partA.ipynb          # Interactive walkthrough for Part A
│  ├─ partB.ipynb          # Interactive walkthrough for Part B
│
├─ partA/
│  ├─ solution_1.py        # Single-image inference (MyCNN)
│  ├─ solution_2.py        # Hyperparameter sweep with MyCNNExtended
│  └─ solution_4.py        # Best-from-scratch training & evaluation
│
├─ partB/
│  └─ solution_3.py        # Progressive fine‑tuning of ResNet‑50
│
├─ utils/
│  ├─ common_utils.py      # Seed setting, config loader, activation lookup
│  ├─ data_utils.py        # Single-image loader & helpers
│  └─ model_utils.py       # Train/validate loops & test evaluation
│
└─ outputs/
   ├─ partA_Q4_best_model.pth
   ├─ partA_Q4_test_predictions_grid.png
   └─ partB_Q3_best_resnet50.pth
```

---

## ⚙️ Installation & Dependencies

1. **Create a Conda or virtualenv** with Python 3.8+.
2. **Install required packages:**
   ```bash
   pip install torch torchvision pytorch-lightning wandb scikit-learn matplotlib pyyaml
   ```
3. **Setup W&B** (if using sweep and logging):
   ```bash
   wandb login  # follow prompt to paste your API key
   ```

---

## 🔧 Configuration

All static parameters—file paths, hyperparameters, early‑stopping settings, W&B project configurations—are managed in **`config/configs.yaml`** under two primary keys:

### Part A (`part_a_configs`)

#### solution_1_configs
- **`data_zip_path`**: Path to the `nature_12K.zip` archive containing the dataset.
- **`extracted_data_dir`**: Directory where the archive is extracted (should contain `inaturalist_12K/`).

#### solution_2_configs
- **`data_root`**: Path to the extracted dataset root (with `train/`, `val/`, `test/`).
- **`gpu_count`**: Number of GPUs for `DataParallel` during the sweep.
- **`wandb_project`**: W&B project name to log sweep runs.
- **`wandb_entity`**: W&B user or team name.
- **`wandb_run_tag`**: Tag added to each sweep run for filtering.
- **`sweep_count`**: Number of hyperparameter trials to execute.
- **`correlation_run_name`**: Name of the W&B run that logs the correlation matrix.

#### solution_4_configs
- **`data_root`**: Path to the extracted dataset.
- **`wandb_project`**: W&B project for best-from-scratch training.
- **`wandb_run_name`**: Name for the final training run in Part A Q4.
- **`output_dir`**: Directory to save checkpoints and plots.
- **`patience`**: Number of epochs without improvement before early-stopping.
- **`perform_early_stopping`**: Boolean to enable or disable early-stopping.

### Part B (`part_b_configs`)

#### solution_3_configs
- **`data_root`**: Path to the extracted dataset root.
- **`wandb_project`**: W&B project for fine‑tuning runs.
- **`wandb_run_name`**: Name for the Part B Q3 run.
- **`output_dir`**: Directory to save the fine‑tuned ResNet checkpoint.
- **`model_config`**: Nested dictionary for fine‑tuning hyperparameters:
  - **`resize_dim`**: Input image size (pixels) for train/validation.
  - **`batch_size`**: Batch size for data loaders.
  - **`val_ratio`**: Fraction of training split for validation.
  - **`seed`**: Random seed for reproducibility.
  - **`learning_rate`**: Base LR for the new head and unfrozen layers.
  - **`epochs`**: Number of training epochs.
  - **`freeze_until_layer`**: Number of ResNet blocks (layer1–layer4) to freeze initially.
  - **`patience`**: Epochs to wait without val‑acc improvement before early-stopping.


- **Part A** (scratch): under `part_a_configs` → `solution_1_configs`, `solution_2_configs`, `solution_4_configs`.
- **Part B** (fine‑tune): under `part_b_configs` → `solution_3_configs`.

A separate `config/sweep_config.yaml` defines the W&B hyperparameter grid for Part A sweep.

---

## 🚀 Running the Code

### 1) Part A: From Scratch

**Solution 1** – Single‑image inference with `MyCNN`:
```bash
python partA/solution_1.py
```

**Solution 2** – Hyperparameter sweep with `MyCNNExtended` (W&B):
```bash
python partA/solution_2.py
```
> This script will create a sweep, run `sweep_train()` for each trial, then produce a correlation matrix in W&B.

**Solution 4** – Train the best scratch CNN & evaluate:
```bash
python partA/solution_4.py
```
> Outputs: checkpoint `outputs/partA_Q4_best_model.pth` and prediction grid image.

---

### 2) Part B: Fine‑Tuning Pre‑trained ResNet‑50

**Solution 3** – Progressive freeze/unfreeze fine‑tuning:
```bash
python partB/solution_3.py
```
> Outputs: checkpoint `outputs/partB_Q3_best_resnet50.pth`, W&B logs, and confusion matrix.

---

## 📊 Evaluation & Results

- **Scratch CNN** (Part A Q4): ~**42.65 %** test accuracy on iNaturalist‑10 subset.
- **Fine‑tuned ResNet‑50** (Part B Q3): ~**90.75 %** test accuracy.

All training/validation curves and confusion matrices are logged to W&B under the specified project names.

---

## 📓 Jupyter Notebooks

For an **interactive walkthrough**, open:
- `notebooks/partA.ipynb` – covers data loading, model definition, training loops for Part A.
- `notebooks/partB.ipynb` – covers data loading, model definition, training loops for Part B.

---

## 📖 Further Notes

- Ensure the dataset is unzipped under `inaturalist_data/nature_12K_extracted/` with `train/`, and `test/` folders.
- Adjust `CUDA_VISIBLE_DEVICES` in each script to match your GPU setup.
- Use `utils/common_utils.py` to modify random seed or activation mappings globally.

---

**Authors:** Kaushik Doddamani, IIT Madras 2024–25

**License:** MIT

