"""Script for running WaveNet from the command line using Lightning CLI."""

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from ml_aos.lightning import WaveNet, DonutLoader

if __name__ == "__main__":


    # setup the CLI
    cli = LightningCLI(
        WaveNet,
        DonutLoader,
    )
