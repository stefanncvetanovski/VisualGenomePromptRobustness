from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification


if __name__ == '__main__':
    project_path = Path(__file__).resolve().parents[1]

    dataset_folder = (
        project_path / 'data' / 'processed' / 'dataset'
    )

    model_folder = (
        project_path / 'models' / 'vit_visual_genome'
    )

    model_name = 'google/vit-base-patch16-224-in21k'

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ])

    train_set = datasets.ImageFolder(
        dataset_folder / 'train',
        transform=train_transform
    )

    validation_set = datasets.ImageFolder(
        dataset_folder / 'validation',
        transform=validation_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    id2label = {}

    for i in range(len(train_set.classes)):
        id2label[i] = train_set.classes[i]

    label2id = {}

    for i in range(len(train_set.classes)):
        label2id[train_set.classes[i]] = i

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(train_set.classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    for parameter in model.vit.parameters():
        parameter.requires_grad = False

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = model.to(device)

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=0.001
    )

    print('Klasi:', train_set.classes)
    print('Train primeri:', len(train_set))
    print('Validation primeri:', len(validation_set))
    print('Ured:', device)

    epochs = 10
    best_validation_accuracy = -1

    for epoch in range(epochs):
        model.train()

        train_loss = 0
        train_correct = 0

        for train_X, train_Y in train_loader:
            train_X = train_X.to(device)
            train_Y = train_Y.to(device)

            optimizer.zero_grad()

            output = model(
                pixel_values=train_X,
                labels=train_Y
            )

            loss = output.loss
            loss.backward()
            optimizer.step()

            predicted_classes = torch.argmax(
                output.logits,
                dim=1
            )

            train_loss += loss.item() * len(train_X)
            train_correct += (
                predicted_classes == train_Y
            ).sum().item()

        train_loss = train_loss / len(train_set)
        train_accuracy = train_correct / len(train_set)

        model.eval()

        validation_loss = 0
        validation_correct = 0

        with torch.no_grad():
            for validation_X, validation_Y in validation_loader:
                validation_X = validation_X.to(device)
                validation_Y = validation_Y.to(device)

                output = model(
                    pixel_values=validation_X,
                    labels=validation_Y
                )

                loss = output.loss

                predicted_classes = torch.argmax(
                    output.logits,
                    dim=1
                )

                validation_loss += loss.item() * len(validation_X)
                validation_correct += (
                    predicted_classes == validation_Y
                ).sum().item()

        validation_loss = validation_loss / len(validation_set)
        validation_accuracy = (
            validation_correct / len(validation_set)
        )

        print()
        print('Epoha:', epoch + 1)
        print('Train loss:', train_loss)
        print('Train accuracy:', train_accuracy)
        print('Validation loss:', validation_loss)
        print('Validation accuracy:', validation_accuracy)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            model.save_pretrained(model_folder)
            image_processor.save_pretrained(model_folder)

            print('Zacuvan e noviot najdobar model.')

    print()
    print('Najdobar validation accuracy:', best_validation_accuracy)
    print('Modelot e zacuvan vo:', model_folder)
