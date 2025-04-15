import os
import sys
import torch
import yaml
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to sys.path if it isn’t already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.common_utils import set_seeds
from src.utils.model_utils import train_one_epoch, validate_one_epoch
from src.data.data_loader import load_inaturalist_train_val
from src.models.implementation import MyCNNExtended

def get_configs(config_filename):
    with open(os.path.join(project_root, "config", config_filename), 'r') as f:
        config = yaml.safe_load(f)
    return config

# ==============================
# Main training function
# ==============================
def sweep_train():
    """
    This function is called by wandb.agent(...) for each sweep run.
    It reads the config from wandb.config, sets up data & model,
    trains and logs results to W&B.
    """
    # Initialize a W&B run so that wandb.config is available.
    wandb.init()
    sweep_config = wandb.config
    static_config = get_configs('configs.yaml')['solution_2_configs']

    # Construct a run name based on various hyperparameters.
    run_name = (
        f"NumFilt: {sweep_config.num_filters}, KSize: {sweep_config.kernel_size}, ActFn: {sweep_config.activation_fn}, "
        f"DenseNrns: {sweep_config.dense_neurons}, FiltOrg: {sweep_config.filter_organization}, DataAug: {sweep_config.data_augmentation}, "
        f"BN: {sweep_config.batch_norm}, DR: {sweep_config.dropout_rate}, LR: {sweep_config.learning_rate}, "
        f"BS: {sweep_config.batch_size}, Epochs: {sweep_config.epochs}, ResizeDim: {sweep_config.resize_dim}"
    )
    wandb.run.name = run_name
    wandb.run.tags = [static_config['wandb_run_tag']]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seeds(42)

    # Build the model using W&B config
    # e.g. config.activation_fn might be "relu"
    act_fn = None
    if sweep_config.activation_fn == "relu":
        act_fn = nn.ReLU
    elif sweep_config.activation_fn == "gelu":
        act_fn = nn.GELU
    elif sweep_config.activation_fn == "silu":
        act_fn = nn.SiLU
    elif sweep_config.activation_fn == "mish":
        act_fn = nn.Mish
    else:
        act_fn = nn.ReLU # fallback

    try:
        model = MyCNNExtended(
            in_channels=3,
            num_filters=sweep_config.num_filters,
            kernel_size=sweep_config.kernel_size,
            activation_fn=act_fn,
            dense_neurons=sweep_config.dense_neurons,
            image_height=sweep_config.resize_dim,
            image_width=sweep_config.resize_dim,
            filter_organization=sweep_config.filter_organization,
            batch_norm=sweep_config.batch_norm,
            dropout_rate=sweep_config.dropout_rate
        ).to(device)

        # Load data: train + val
        train_dir = os.path.join(static_config['data_root'], "train")
        train_dataset, val_dataset, class_names = load_inaturalist_train_val(
            data_dir=train_dir,
            val_ratio=0.2,
            seed=42,
            augment=sweep_config.data_augmentation,
            resize_dim=sweep_config.resize_dim
        )

        train_loader = DataLoader(train_dataset, batch_size=sweep_config.batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset,   batch_size=sweep_config.batch_size, shuffle=False, num_workers=4)

        # Optimizer & Loss
        optimizer = optim.Adam(model.parameters(), lr=sweep_config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Train loop
        for epoch in range(sweep_config.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            print(f"[Epoch {epoch+1}/{sweep_config.epochs}] "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    except torch.cuda.OutOfMemoryError as e:
        print("CUDA out of memory error encountered. Skipping current run.")
        wandb.log({"error": "CUDA out of memory encountered"})
        # Clear GPU cache
        torch.cuda.empty_cache()
    finally:
        wandb.finish()  # ensure that wandb finishes regardless of errors


def main():
    """
    This function:
      1) Defines the sweep_config as a Python dict.
      2) Creates the sweep via `wandb.sweep(...)`.
      3) Starts the W&B agent with `wandb.agent(...)`, calling `sweep_train()`.
    """
    # Load static config from YAML file
    static_config = get_configs('configs.yaml')['solution_2_configs']
    # Load sweep config from YAML file
    sweep_config = get_configs('sweep_config.yaml')

    # W&B sweep
    sweep_id = wandb.sweep(sweep_config, project=static_config["wandb_project"])
    wandb.agent(sweep_id, function=sweep_train, count=static_config["sweep_count"])


if __name__ == "__main__":
    main()
