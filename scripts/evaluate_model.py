from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification


if __name__ == '__main__':
    project_path = Path(__file__).resolve().parents[1]

    test_folder = (
        project_path / 'data' / 'processed' / 'dataset' / 'test'
    )

    model_folder = (
        project_path / 'models' / 'vit_visual_genome'
    )

    output_folder = project_path / 'outputs'
    output_folder.mkdir(parents=True, exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained(model_folder)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ])

    test_set = datasets.ImageFolder(
        test_folder,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_folder
    )

    model = model.to(device)
    model.eval()

    test_correct = 0

    real_classes = []
    predicted_classes = []

    with torch.no_grad():
        for test_X, test_Y in test_loader:
            test_X = test_X.to(device)
            test_Y = test_Y.to(device)

            output = model(pixel_values=test_X)

            predictions = torch.argmax(
                output.logits,
                dim=1
            )

            test_correct += (
                predictions == test_Y
            ).sum().item()

            real_classes.extend(
                test_Y.cpu().tolist()
            )

            predicted_classes.extend(
                predictions.cpu().tolist()
            )

    test_accuracy = test_correct / len(test_set)

    print('Test primeri:', len(test_set))
    print('Test accuracy:', test_accuracy)
    print()

    print(
        classification_report(
            real_classes,
            predicted_classes,
            target_names=test_set.classes,
            digits=4
        )
    )

    matrix = confusion_matrix(
        real_classes,
        predicted_classes
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=test_set.classes
    )

    figure, axes = plt.subplots(figsize=(10, 8))

    display.plot(
        ax=axes,
        cmap='Blues',
        values_format='d'
    )

    plt.xticks(rotation=45)
    figure.tight_layout()

    confusion_matrix_path = output_folder / 'confusion_matrix.png'

    figure.savefig(confusion_matrix_path)

    print('Confusion matrix e zacuvana vo:', confusion_matrix_path)
