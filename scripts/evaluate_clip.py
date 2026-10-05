import random
from pathlib import Path

import torch
from torchvision import datasets
from transformers import CLIPModel, CLIPProcessor


if __name__ == '__main__':
    project_path = Path(__file__).resolve().parents[1]

    test_folder = (
        project_path / 'data' / 'processed' / 'dataset' / 'test'
    )

    output_folder = project_path / 'outputs'
    output_folder.mkdir(parents=True, exist_ok=True)

    model_name = 'openai/clip-vit-base-patch32'

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = model.to(device)
    model.eval()

    test_set = datasets.ImageFolder(test_folder)

    text_descriptions = [
        'a photo of a ' + label
        for label in test_set.classes
    ]

    random.seed(0)

    clip_correct = 0
    misinformation_correct = 0

    for i in range(len(test_set)):
        image, real_class = test_set[i]

        inputs = processor(
            text=text_descriptions,
            images=image,
            return_tensors='pt',
            padding=True
        )

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            output = model(**inputs)

        scores = output.logits_per_image[0]

        predicted_class = torch.argmax(scores).item()

        if predicted_class == real_class:
            clip_correct += 1

        false_classes = [
            class_index
            for class_index in range(len(test_set.classes))
            if class_index != real_class
        ]

        false_class = random.choice(false_classes)

        true_score = scores[real_class].item()
        false_score = scores[false_class].item()

        if true_score > false_score:
            misinformation_correct += 1

        if (i + 1) % 50 == 0:
            print(
                'Obraboteni primeri:',
                i + 1,
                '/',
                len(test_set)
            )

    clip_accuracy = clip_correct / len(test_set)

    misinformation_accuracy = (
        misinformation_correct / len(test_set)
    )

    print()
    print('CLIP zero-shot accuracy:', clip_accuracy)
    print(
        'Misinformation detection accuracy:',
        misinformation_accuracy
    )

    results_path = output_folder / 'clip_results.txt'

    with open(results_path, 'w', encoding='utf-8') as file:
        file.write(
            'CLIP zero-shot accuracy: ' +
            str(clip_accuracy) +
            '\n'
        )

        file.write(
            'Misinformation detection accuracy: ' +
            str(misinformation_accuracy) +
            '\n'
        )

    print('Rezultatite se zacuvani vo:', results_path)
