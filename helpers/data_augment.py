import os
from os.path import relpath, join as join_path
from random import sample
from typing import List

import PIL.Image
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxFormat
from tqdm import trange

from custom_transform.glare import Glare
from custom_transform.overlay import Overlay

perspective_transform = v2.RandomPerspective(p=1.0)
rotate_transform = v2.RandomRotation(degrees=180)
affine_transform = v2.RandomAffine(degrees=180, translate=(0.2, 0.2), scale=(0.5, 1.5), shear=30)
elastic_transform = v2.ElasticTransform(alpha=120.0)
resized_crop_transform = v2.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0))
color_jitter_transform = v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
gaussian_blur_transform = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
solarize_transform = v2.RandomSolarize(threshold=128.0)
posterize_transform = v2.RandomPosterize(bits=4)
horizontal_flip_transform = v2.RandomHorizontalFlip(p=1.0)
glare_transform = Glare()
len_smear_transform = Overlay(alpha=0.5)
fog_overlay_transform = Overlay(color=255, alpha=0.2)


def transform_origin_image(img: PIL.Image.Image, bboxes: List[List[int]]):
    transform_fns = [
        perspective_transform,
        rotate_transform,
        affine_transform,
        # elastic_transform,
        resized_crop_transform,
        color_jitter_transform,
        gaussian_blur_transform,
        solarize_transform,
        posterize_transform,
        horizontal_flip_transform,
        glare_transform,
        len_smear_transform,
        fog_overlay_transform,
    ]

    selected_fns = sample(transform_fns, 3)
    return v2.Compose(selected_fns)((img, bboxes))


def prepare_augmented_dataset(origin_dataset: datasets.WIDERFace, stored_location: str, split: str):
    # for each sample in the origin_dataset, apply transform_origin_image
    # and store the transformed image in
    # (stored_location)/AUG_WIDER_(split)/images/(image_prefix)/(image_name)_aug_(index).jpg
    # and store the transformed bboxes in (stored_location)/AUG_wider_face_split/aug_wider_face_(split)_bbx_gt.txt
    # in the format:
    # (image_name)
    # (number of bboxes)
    # (x1) (y1) (w) (h) (is_aug)

    # Prepare a list to store the labels
    temp = dict()

    # Check if folder exists, if not, create it
    if not os.path.exists(f'{stored_location}/AUG_WIDER_{split}/images'):
        os.makedirs(f'{stored_location}/AUG_WIDER_{split}/images')

    for i in trange(len(origin_dataset)):
        img, bboxes = origin_dataset[i]
        img_name = img.filename
        img_prefix = img_name.split('/')[-2]
        img_name = img_name.split('/')[-1]
        raw_img_name = img_name.split('.')[0]

        # Check if folder exists, if not, create it
        if not os.path.exists(f'{stored_location}/AUG_WIDER_{split}/images/{img_prefix}'):
            os.makedirs(f'{stored_location}/AUG_WIDER_{split}/images/{img_prefix}')

        # Store the original image
        img.save(f'{stored_location}/AUG_WIDER_{split}/images/{img_prefix}/{img_name}', 'jpeg')
        temp[f'{img_prefix}/{img_name}'] = bboxes

        for j in range(3):
            aug_img, aug_bboxes = transform_origin_image(img, bboxes)
            if not isinstance(aug_img, PIL.Image.Image):
                aug_img = v2.ToPILImage()(aug_img)

            aug_img.save(f'{stored_location}/AUG_WIDER_{split}/images/{img_prefix}/{raw_img_name}_aug_{j}.jpg', 'jpeg')
            temp[f'{img_prefix}/{raw_img_name}_aug_{j}.jpg'] = aug_bboxes

    # Store the labels
    if not os.path.exists(f'{stored_location}/AUG_wider_face_split'):
        os.makedirs(f'{stored_location}/AUG_wider_face_split')

    with open(f'{stored_location}/AUG_wider_face_split/aug_wider_face_{split}_bbx_gt.txt', 'w') as f:
        for fname, bboxes in temp.items():
            f.write(fname + '\n')
            f.write(str(len(bboxes["bbox"])) + '\n')
            bboxes = v2.ConvertBoundingBoxFormat(format=BoundingBoxFormat.XYWH)(bboxes)
            for bbox in bboxes["bbox"]:
                # Convert tensor to x y h w
                bbox = bbox.tolist()
                f.write(' '.join([str(x) for x in bbox]) + '\n')

    return origin_dataset[0]


def generate_img_filelist_txt(folder: str, destination: str):
    if not os.path.exists(folder):
        raise FileNotFoundError(f'{folder} does not exist')

    # Check destination parent folder exists
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))

    # Read all images in folder
    img_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg'):
                # Append the difference between folder path and file path
                img_list.append(relpath(join_path(root, file), folder))

    # Write to destination
    with open(destination, 'w') as f:
        for img in img_list:
            f.write(img + '\n')
