import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from .sr_dataset import SRDataset


class SRWiderfaceDataModule(pl.LightningDataModule):

    def __init__(self, root: str):
        super().__init__()
        self.root = root

    def load_dataset(self, split: str):
        return SRDataset(
            root=self.root,
            split=split,
        )

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_dataset = self.load_dataset(split='train')
            self.val_dataset = self.load_dataset(split='val')
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=16,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=16,
            drop_last=True,
        )
