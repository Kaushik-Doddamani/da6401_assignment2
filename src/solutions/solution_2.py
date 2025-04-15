import os
import sys
import torch
import yaml
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from wandb import Api

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

# ------------------------------------------------------------------------------------
# Generate a correlation table after the sweep finishes, logging it to a new W&B run
# ------------------------------------------------------------------------------------
def  generate_correlation_table(static_config, sweep_id):
    """
    Connects to the W&B API, retrieves all runs for the given sweep_id that also
    contain the specified tag, gathers numeric hyperparameters and final metrics,
    computes correlations, and logs the correlation matrix as a wandb.Table in a new run.
    
    :param project: (str) W&B project name (e.g., "inat_sweep_demo")
    :param sweep_id: (str) The unique ID of the sweep.
    :param run_tag:  (str) A tag used to mark sweep runs (if provided, only runs having
                          that tag will be considered).
    """
    api = Api()
    project = static_config["wandb_project"]
    run_tag = static_config["wandb_run_tag"]
    corr_run_name = static_config["correlation_run_name"]
    
    # Build filter: always filter by sweep_id.
    filters = {"sweep": sweep_id}
    if run_tag is not None:
        filters["tags"] = {"$in": [run_tag]}
    
    # Query runs by project
    runs = api.runs(f"{project}", filters=filters)
    if not runs:
        print(f"No runs found for sweep_id: {sweep_id} with tag: {run_tag}")
        return

    records = []
    for run in runs:
        # Get final metrics from run.summary (e.g., val_accuracy, val_loss)
        val_acc = run.summary.get("val_accuracy", None)
        val_loss = run.summary.get("val_loss", None)
        
        # Build dict with hyperparameters and metrics
        row = {
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "num_filters":          run.config.get("num_filters"),
            "kernel_size":          run.config.get("kernel_size"),
            "activation_fn":        run.config.get("activation_fn"),
            "dense_neurons":        run.config.get("dense_neurons"),
            "filter_organization":  run.config.get("filter_organization"),
            "data_augmentation":    run.config.get("data_augmentation"),
            "batch_norm":           run.config.get("batch_norm"),
            "dropout_rate":         run.config.get("dropout_rate"),
            "learning_rate":        run.config.get("learning_rate"),
            "batch_size":           run.config.get("batch_size"),
            "epochs":               run.config.get("epochs"),
            "resize_dim":           run.config.get("resize_dim"),
        }
        records.append(row)
    
    df = pd.DataFrame(records)
    if df.empty:
        print("No data collected from runs. Exiting.")
        return

    # Correlation can only be computed on numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns to compute correlation on.")
        return

    # Compute the correlation matrix
    corr_matrix = numeric_df.corr()

    # Create a new W&B run to log the correlation table
    wandb.init(project=project, name=corr_run_name)
    
    corr_cols = corr_matrix.columns.tolist()  # e.g. ["val_accuracy", "val_loss", "num_filters", ...]
    # The first column in our table is the row label (metric/param name), 
    # then the rest are the correlation values corresponding to each col.
    table_columns = [""] + corr_cols  # The first column is blank (for row label)
    correlation_table = wandb.Table(columns=table_columns)

    # For each row in corr_matrix, add a row to our wandb.Table
    for row_name in corr_cols:
        row_data = corr_matrix.loc[row_name, :].values.tolist()  # correlation values for row_name
        # row: row label + the correlation values
        correlation_table.add_data(row_name, *row_data)

    # Log the table so we see the row labels properly
    wandb.log({"hyperparam_correlation_matrix": correlation_table})

    print("=== Correlation Matrix ===")
    print(corr_matrix)
    wandb.finish()


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

    # Determine activation function
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
    1) Reads static config + sweep config from YAML.
    2) Creates the sweep.
    3) Runs the W&B agent for the desired number of runs.
    4) After the sweep completes, we query all runs, build a correlation table,
       and log it in a new run called 'correlation_analysis'.
    """
    # Load static config from YAML file
    static_config = get_configs('configs.yaml')['solution_2_configs']
    # Load sweep config from YAML file
    sweep_config = get_configs('sweep_config.yaml')

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project=static_config["wandb_project"])
    # Run sweep experiments
    wandb.agent(sweep_id, function=sweep_train, count=static_config["sweep_count"])

    # Once the agent finishes count=... runs, we generate the correlation table
    # Programmatically create & log correlation table
    generate_correlation_table(
        static_config=static_config,
        sweep_id=sweep_id
    )


if __name__ == "__main__":
    main()
