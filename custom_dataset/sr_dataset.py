import os
import re
from typing import Optional, Callable

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision.transforms import v2
from torchvision import tv_tensors


class SRDataset(VisionDataset):
    BASE_FOLDER = "aug_widerface"
    FILELISTS = {
        "train": "aug_wider_face_train_filelist.txt",
        "val": "aug_wider_face_val_filelist.txt",
        "test": "aug_wider_face_test_filelist.txt",
    }

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None:
        super().__init__(
            root=os.path.join(root, self.BASE_FOLDER), transform=transform, target_transform=target_transform
        )
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))
        self.img_info = []
        self.prepare_lq_image()

    def __getitem__(self, index):
        sample_path = self.img_info[index]
        hq_image = Image.open(sample_path['hq'])
        pil_image = Image.open(sample_path['lq'])
        w, h = pil_image.size
        # Downsample the image
        scale = (0.8 - 8) * torch.rand() + 8
        pil_image = pil_image.resize((int(w // scale), int(h // scale)), resample=Image.BILINEAR)

        # Add Gaussian noise
        tv_img = tv_tensors.Image(pil_image)
        tv_img = v2.GaussianBlur(kernel_size=3, sigma=(0, 20))(tv_img)

        # Rescale the image
        pil_image = v2.ToPILImage()(tv_img)
        pil_image = pil_image.resize((w, h), resample=Image.BILINEAR)

        return dict(
            lq=pil_image,
            hq=hq_image,
            scale=scale,
        )

    def prepare_lq_image(self):
        # Load the file list
        filelist_path = os.path.join(self.root, self.FILELISTS[self.split])
        with open(filelist_path, 'r') as f:
            filelist = f.readlines()

        for file in filelist:
            file = file.strip()

            # if file ends with aug_{index}.jpg, then it is a transformed image
            # of the original image with the same name
            # if file does not end with aug_{index}.jpg, then it is the original image

            # Check if file is a transformed image
            if re.match(r'.+aug_\d+\.jpg', file):
                # Get the original image name
                original_file = re.sub(r'_aug_\d+\.jpg', '.jpg', file)
                # Get the original image path
                original_file_path = os.path.join(self.root, 'AUG_WIDER_' + self.split, 'images', original_file)
                lq_file_path = os.path.join(self.root, 'AUG_WIDER_' + self.split, 'images', file)
                # Save to self.img_info
                self.img_info.append({
                    'hq': original_file_path,
                    'lq': lq_file_path,
                })
