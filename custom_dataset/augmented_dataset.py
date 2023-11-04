import os
from typing import Optional, Callable, Dict, Union, List
import torch

from os.path import aspath, expanduser

from PIL.Image import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


class AugmentedDataset(VisionDataset):
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

        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []
        if self.split in ("train", "val"):
            self.parse_train_val_annotations_file()
        else:
            self.parse_test_annotations_file()

    def __getitem__(self, index):
        img = Image.open(self.img_info[index]["img_path"])

        if self.transform is not None:
            img = self.transform(img)

        target = None if self.split == "test" else self.img_info[index]["annotations"]
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
        filename = "aug_wider_face_train_bbx_gt.txt" if self.split == "train" else "aug_wider_face_val_bbx_gt.txt"
        filepath = os.path.join(self.root, "AUG_wider_face_split", filename)

        with open(filepath) as f:
            lines = f.readlines()
            file_name_line, num_boxes_line, box_annotation_line = True, False, False
            num_boxes, box_counter = 0, 0
            labels = []

            for line in lines:
                line = line.rstrip()
                if file_name_line:
                    img_path = os.path.join(self.root, "AUG_WIDER_" + self.split, "images", line)
                    img_path = aspath(expanduser(img_path))
                    file_name_line, num_boxes_line = False, True
                elif num_boxes_line:
                    num_boxes = int(line)
                    num_boxes_line, box_annotation_line = False, True
                elif box_annotation_line:
                    box_counter += 1
                    line_split = line.split(" ")
                    line_values = [int(x) for x in line_split]
                    labels.append(line_values)
                    if box_counter >= num_boxes:
                        box_annotation_line, file_name_line = False, True
                        labels_tensor = torch.tensor(labels)
                        self.img_info.append(
                            {"img_path": img_path, "annotations": {"bbox": labels_tensor[: 0:4].clone()}})
                        box_counter = 0
                        labels.clear()
                else:
                    raise RuntimeError(f"Error while parsing annotations file {filepath}")

    def parse_test_annotations_file(self) -> None:
        filepath = os.path.join(self.root, "AUG_wider_face_split", "aug_wider_face_test_filelist.txt")
        filepath = aspath(expanduser(filepath))
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                img_path = os.path.join(self.root, "AUG_WIDER_test", "images", line)
                img_path = aspath(expanduser(img_path))
                self.img_info.append({"img_path": img_path})
