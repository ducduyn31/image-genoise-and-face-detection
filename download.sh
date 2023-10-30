#!/usr/bin/env bash

# List the datasets to download
declare -A datasets

datasets["WIDER_face"]="https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip"

mkdir -p "temp"

# Download the datasets
for dataset in "${!datasets[@]}"; do
    url=${datasets[$dataset]}
    echo "Downloading $dataset"
    wget -q --show-progress $url -O temp/$dataset.zip
done

# Extract the datasets to data/[Dataset Name]
for dataset in "${!datasets[@]}"; do
    echo "Extracting $dataset"
    unzip -qj temp/$dataset.zip -d data/$dataset
    rm temp/$dataset.zip
done
