import json
import os
from os.path import abspath, expanduser
from typing import Optional, Callable, Dict, Union, List

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision.tv_tensors._dataset_wrapper import WRAPPER_FACTORIES


class CelebASpoofDataset(VisionDataset):
    BASE_FOLDER = "celebaspoof"

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
            raise RuntimeError("Dataset not found or corrupted.")

        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []
        if self.split in ("train", "val"):
            self.parse_train_val_annotations_file()
        else:
            self.parse_test_annotations_file()

    def __getitem__(self, index):
        img = Image.open(self.img_info[index]["img_path"])

        # Crop the image to the bounding box
        if self.split in ("train", "val"):
            bbox = self.img_info[index]["annotations"]["bbox"]
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        if self.transform is not None:
            img = self.transform(img)

        target = None if self.split == "test" else self.img_info[index]["annotations"]["is_spoof"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_info)

    def extra_repr(self) -> str:
        lines = ["Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)

    def _check_integrity(self) -> bool:
        return True

    def parse_train_val_annotations_file(self) -> None:
        filename = "celeb_a_spoof_train_annotations.json" if self.split == "train" else "celeb_a_spoof_val_annotations.json"
        filepath = os.path.join(self.root, "metas/intra_test", filename)

        with open(filepath) as f:
            labels = json.load(f)

        for img_name, values in labels.items():
            img_path = os.path.join(self.root, "images", img_name)
            img_path = abspath(expanduser(img_path))

            with open(img_path.replace(".jpg", "_BB.txt")) as bbox_f:
                bbox_line = bbox_f.readline()
                x, y, w, h, _ = bbox_line.strip().split(" ")

            self.img_info.append({
                "img_path": img_path,
                "annotations": {
                    "is_spoof": torch.Tensor(0 if values[43] == "live" else 1),
                    "bbox": torch.tensor([x, y, w, h]),
                }
            })

    def parse_test_annotations_file(self) -> None:
        filename = "celeb_a_spoof_test_filelist.txt"
        filepath = os.path.join(self.root, "metas/intra_test", filename)

        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                img_name = line.strip()
                img_path = os.path.join(self.root, "images", img_name)
                img_path = abspath(expanduser(img_path))

                self.img_info.append({
                    "img_path": img_path,
                })


def parse_target_keys(target_keys, *, available, default):
    if target_keys is None:
        target_keys = default
    if target_keys == "all":
        target_keys = available
    else:
        target_keys = set(target_keys)
        extra = target_keys - available
        if extra:
            raise ValueError(f"Target keys {sorted(extra)} are not available")

    return target_keys


@WRAPPER_FACTORIES.register(CelebASpoofDataset)
def ds_wrapper(dataset, target_keys):
    target_keys = parse_target_keys(
        target_keys,
        available={
            "is_spoof",
        },
        default="all",
    )

    def wrapper(idx, sample):
        image, target = sample

        if target is None:
            return image, target

        target = {key: target[key] for key in target_keys}

        return image, target

    return wrapper
