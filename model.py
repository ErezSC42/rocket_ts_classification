import torch
import torch.nn as nn
import pytorch_lightning as pl


class RocketNet(pl.LightningModule):
    def __init__(
            self,
            n_classes: int,
            kernel_count: int,
            kernel_config: dict):
        self.n_classes = n_classes
        self.feature_dim = 2 * kernel_count
        # linear classifier
        self._fc = nn.Linear(self.feature_dim, n_classes)
        self._max_pooling = nn.MaxPool1d(self.feature_dim)  # check correct dim

        # TODO generate the random kernels
        pass

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass
