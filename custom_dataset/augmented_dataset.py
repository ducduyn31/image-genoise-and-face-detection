import os
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg


class AugmentedDataset(Dataset):
    BASE_FOLDER = "aug_widerface"

    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None:
        super().__init__(
            root=os.path.join(root, self.BASE_FOLDER), transform=transform, target_transform=target_transform
        )

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download and prepare it")

    def __getitem__(self, index):
        img = Image.open(self.)

    def __len__(self):
        return len(self.dataset)

    def _check_integrity(self) -> bool:
        return True
