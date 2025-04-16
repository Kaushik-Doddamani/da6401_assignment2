import os
import sys
from torch import nn, optim
import pytorch_lightning as pl

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to sys.path if it isn’t already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.implementation import MyCNNExtended


class LitInatModel(pl.LightningModule):
    """ Wraps MyCNN in a LightningModule for multi-GPU & logging. """

    def __init__(
            self,
            in_channels=3,
            num_filters=32,
            kernel_size=3,
            activation_fn="relu",
            dense_neurons=128,
            filter_organization="same",
            data_augmentation=True,
            batch_norm=False,
            dropout_rate=0.0,
            learning_rate=1e-3,
            batch_size=32,
            epochs=5,
            resize_dim=224):
        super().__init__()
        # Save hyperparams to check them later
        self.save_hyperparameters()

        # Decide activation function
        if activation_fn.lower() == "relu":
            act_fn = nn.ReLU
        elif activation_fn.lower() == "gelu":
            act_fn = nn.GELU
        elif activation_fn.lower() == "silu":
            act_fn = nn.SiLU
        elif activation_fn.lower() == "mish":
            act_fn = nn.Mish
        else:
            act_fn = nn.ReLU

        # Build CNN
        self.model = MyCNNExtended(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            activation_fn=act_fn,
            dense_neurons=dense_neurons,
            image_height=resize_dim,
            image_width=resize_dim,
            filter_organization=filter_organization,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
