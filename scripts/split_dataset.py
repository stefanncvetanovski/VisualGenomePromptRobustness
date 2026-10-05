import random
import shutil
from pathlib import Path


if __name__ == '__main__':
    project_path = Path(__file__).resolve().parents[1]

    crops_folder = (
        project_path / 'data' / 'processed' / 'crops'
    )

    dataset_folder = (
        project_path / 'data' / 'processed' / 'dataset'
    )

    labels = [
        'car',
        'table',
        'chair',
        'dog',
        'cat',
        'bicycle',
        'bus',
        'horse'
    ]

    random.seed(0)

    for label in labels:
        label_folder = crops_folder / label

        if not label_folder.is_dir():
            continue

        images = [
            image
            for image in label_folder.iterdir()
            if image.suffix.lower() == '.jpg'
        ]

        random.shuffle(images)

        train_end = int(0.70 * len(images))
        validation_end = int(0.85 * len(images))

        train_set = images[:train_end]
        validation_set = images[train_end:validation_end]
        test_set = images[validation_end:]

        train_folder = dataset_folder / 'train' / label
        validation_folder = dataset_folder / 'validation' / label
        test_folder = dataset_folder / 'test' / label

        train_folder.mkdir(parents=True, exist_ok=True)
        validation_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        for image in train_set:
            destination = train_folder / image.name
            shutil.copy(image, destination)

        for image in validation_set:
            destination = validation_folder / image.name
            shutil.copy(image, destination)

        for image in test_set:
            destination = test_folder / image.name
            shutil.copy(image, destination)

        print(
            label,
            '- train:', len(train_set),
            '- validation:', len(validation_set),
            '- test:', len(test_set)
        )