"""Wrapping everything for WaveNet in Pytorch Lightning."""

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import wandb
from ml_aos.dataloader import Donuts
from ml_aos.wave_net import WaveNet as TorchWaveNet


class DonutLoader(pl.LightningDataModule):
    """Pytorch Lightning wrapper for the simulated Donuts DataSet."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 16,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the simulated Donuts data.

        Parameters
        ----------
        batch_size: int, default=64
            The batch size for SGD.
        num_workers: int, default=16
            The number of workers for parallel loading of batches.
        persistent_workers: bool, default=True
            Whether to shutdown worker processes after dataset is consumed once
        pin_memory: bool, default=True
            Whether to automatically put data in pinned memory (recommended
            whenever using a GPU).
        **kwargs
            See the keyword arguments in the Donuts class.
        """
        super().__init__()
        self.save_hyperparameters()

    def _build_loader(
        self, mode: str, shuffle: bool = False, drop_last: bool = True
    ) -> DataLoader:
        """Build a DataLoader"""
        return DataLoader(
            Donuts(mode=mode, **self.hparams),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader("val", drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self._build_loader("test", drop_last=False)


class WaveNet(TorchWaveNet, pl.LightningModule):
    """Pytorch Lightning wrapper for WaveNet."""

    def __init__(self, n_meta_layers: int = 3) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        n_meta_layers: int, default=3
            Number of layers in the MetaNet inside the WaveNet. These
            are the linear layers that map image features plus field
            position to Zernike coefficients.
        """
        # set up the WaveNet implemented in torch,
        # as well as the LightningModule boilerplate
        super().__init__(n_meta_layers=n_meta_layers)

        # save the hyperparams in the log
        self.save_hyperparameters()

    def _predict(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions for a batch of donuts."""
        # unpack the data
        img = batch["image"]
        dof_true = batch["dof"]
        intra = batch["intrafocal"]

        # predict the zernikes
        dof_pred = self(img, intra)

        return dof_pred, dof_true

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Calculate the loss of the training step."""
        # calculate the MSE for the batch
        dof_pred, dof_true = self._predict(batch)
        mse = torch.mean((dof_pred - dof_true)**2, axis=1, keepdim=True)


        # log the mean rmse
        self.log("train_rmse", torch.sqrt(mse).mean())

        # loss = mean mse
        loss = mse.mean()
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, ...]:
        """Perform validation step."""
        # calculate the MSE for the validation sample
        dof_pred, dof_true = self._predict(batch)
        mse = torch.mean((dof_pred - dof_true)**2, axis=1, keepdim=True)

        # log the mean rmse
        self.log("val_rmse", torch.sqrt(mse).mean())

        # log the loss
        self.log("val_loss", mse.mean())

        val_outputs = mse

        return val_outputs

    def validation_epoch_end(self, val_outputs: torch.Tensor) -> None:
        """Compute metrics for the whole validation epoch."""

        mse = torch.stack(val_outputs)
        self.log("val_loss", mse.mean())
        self.log("val_rmse", torch.sqrt(mse).mean())