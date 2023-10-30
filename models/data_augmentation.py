from datasets import load_dataset_builder

if __name__ == '__main__':
    # Load dataset builder for WiderFace
    dataset_builder = load_dataset_builder('wider_face')

    # Print description of dataset builder
    print(dataset_builder.info)
