import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import v2

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

    def collate_fn(self, batch):
        hq = [b['hq'] for b in batch]
        lq = [b['lq'] for b in batch]
        max_res_w = max([b.shape[2] for b in hq] + [b.shape[2] for b in lq])
        max_res_h = max([b.shape[1] for b in hq] + [b.shape[1] for b in lq])
        # pad to max res using CenterCrop
        hq = torch.stack([v2.CenterCrop([max_res_h, max_res_w])(b) for b in hq])
        lq = torch.stack([v2.CenterCrop([max_res_h, max_res_w])(b) for b in lq])

        return dict(
            lq=lq,
            hq=hq,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=16,
            shuffle=True,
            num_workers=16,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=16,
            shuffle=False,
            num_workers=16,
            drop_last=True,
        )
