import json
import random
from collections import defaultdict
from pathlib import Path


if __name__ == '__main__':
    project_path = Path(__file__).resolve().parents[1]

    image_data_path = project_path / 'data' / 'raw' / 'image_data.json'
    objects_path = project_path / 'data' / 'raw' / 'objects.json'
    output_path = project_path / 'data' / 'processed' / 'selected_objects.json'

    target_classes = [
        'car',
        'table',
        'chair',
        'dog',
        'cat',
        'bicycle',
        'bus',
        'horse'
    ]

    samples_per_class = 400

    with open(image_data_path, encoding='utf-8') as file:
        images = json.load(file)

    with open(objects_path, encoding='utf-8') as file:
        objects_data = json.load(file)

    image_urls = {}

    for image in images:
        image_urls[image['image_id']] = image['url']

    candidates = defaultdict(list)

    for image in objects_data:
        image_id = image['image_id']

        for object_item in image['objects']:
            if len(object_item['names']) == 0:
                continue

            object_name = object_item['names'][0].strip().lower()

            if object_name not in target_classes:
                continue

            if object_item['w'] < 80 or object_item['h'] < 80:
                continue

            candidates[object_name].append({
                'image_id': image_id,
                'url': image_urls[image_id],
                'label': object_name,
                'x': object_item['x'],
                'y': object_item['y'],
                'w': object_item['w'],
                'h': object_item['h']
            })

    random.seed(42)

    selected_objects = []

    for object_name in target_classes:
        random.shuffle(candidates[object_name])

        selected_objects.extend(
            candidates[object_name][:samples_per_class]
        )

        print(object_name, ':', len(candidates[object_name][:samples_per_class]))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(selected_objects, file, indent=2)

    print()
    print('Total selected objects:', len(selected_objects))
    print('Saved to:', output_path)