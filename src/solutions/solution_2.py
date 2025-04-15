import os
import sys
import argparse
import torch
import yaml
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to sys.path if it isn’t already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.wrapper import LitInatModel
from src.data.data_loader import InatDataModule

def get_configs(config_filename):
    with open(os.path.join(project_root, "config", config_filename), 'r') as f:
        config = yaml.safe_load(f)
    return config

def sweep_train():
    """
    This function is called by `wandb.agent(sweep_id, function=sweep_train, ...)`
    for each hyperparameter config. We read `wandb.config`, create data+model,
    optionally compile, train, then finish.
    """
    wandb.init()  # start a W&B run
    sweep_config = wandb.config
    static_config = get_configs('configs.yaml')['solution_2_configs']

    # Construct a run name based on various hyperparameters.
    run_name = (f"NumFilt: {sweep_config.num_filters}, KSize: {sweep_config.kernel_size}, ActFn: {sweep_config.activation_fn}, "
                f"DenseNrns: {sweep_config.dense_neurons}, FiltOrg: {sweep_config.filter_organization}, DataAug: {sweep_config.data_augmentation}, "
                f"BN: {sweep_config.batch_norm}, DR: {sweep_config.dropout_rate}, LR: {sweep_config.learning_rate}, "
                f"BS: {sweep_config.batch_size}, Epochs: {sweep_config.epochs}, ResizeDim: {sweep_config.resize_dim}")
    wandb.run.name = run_name
    wandb.run.tags = [static_config['wandb_run_tag']]

    # W&B Logger
    wandb_logger = WandbLogger(project=static_config['wandb_project'])

    # Create the data module with the sweep's hyperparams
    dm = InatDataModule(
        data_dir = os.path.join(static_config['data_root'], "train"),
        val_ratio=0.2,
        seed=42,
        augment=sweep_config.data_augmentation,
        resize_dim=sweep_config.resize_dim,
        batch_size=sweep_config.batch_size,
        num_workers=4
    )

    # Create the model with the sweep config
    model = LitInatModel(
        in_channels=3,
        num_filters=sweep_config.num_filters,
        kernel_size=sweep_config.kernel_size,
        activation_fn=sweep_config.activation_fn,
        dense_neurons=sweep_config.dense_neurons,
        filter_organization=sweep_config.filter_organization,
        data_augmentation=sweep_config.data_augmentation,
        batch_norm=sweep_config.batch_norm,
        dropout_rate=sweep_config.dropout_rate,
        learning_rate=sweep_config.learning_rate,
        batch_size=sweep_config.batch_size,
        epochs=sweep_config.epochs,
        resize_dim=sweep_config.resize_dim
    )

    # Checkpoint callback
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_acc",
    #     mode="max",
    #     filename="inat-{epoch:02d}-{val_acc:.2f}",
    #     dirpath="./checkpoints_sweep"
    # )

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,                 # or more if you want multi-GPU per run
        strategy="ddp",            # Distributed Data Parallel
        max_epochs=sweep_config.epochs,
        # callbacks=[checkpoint_callback],
        precision=16
    )

    # Fit
    trainer.fit(model, dm)

    # Finish the W&B run
    wandb.finish()



def main():
    """
    This function:
      1) Defines the sweep_config as a Python dict.
      2) Creates the sweep via `wandb.sweep(...)`.
      3) Starts the W&B agent with `wandb.agent(...)`, calling `sweep_train()`.
    """
    static_config = get_configs('configs.yaml')['solution_2_configs']
    sweep_config = get_configs('sweep_config.yaml')

    # Create the sweep in Python
    sweep_id = wandb.sweep(sweep_config, project=static_config['wandb_project'])

    # Launch the agent, specifying the function to call for each run
    # 'count=5' means run 5 different random combos. Omit or adjust as needed.
    wandb.agent(sweep_id, function=sweep_train, count=static_config['sweep_count'])


if __name__ == "__main__":
    main()
