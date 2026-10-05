import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    project_path = Path(__file__).resolve().parents[1]

    selected_path = project_path / 'data' / 'processed' / 'selected_objects.json'
    output_folder = project_path / 'data' / 'processed' / 'crops'
    failed_path = project_path / 'data' / 'processed' / 'failed_downloads.json'

    with open(selected_path, encoding='utf-8') as file:
        selected_objects = json.load(file)

    failed_downloads = []
    downloaded_count = 0
    skipped_count = 0

    for index, selected_object in enumerate(tqdm(selected_objects)):
        label = selected_object['label']
        image_id = selected_object['image_id']

        label_folder = output_folder / label
        label_folder.mkdir(parents=True, exist_ok=True)

        output_path = label_folder / f'{index}_{image_id}.jpg'

        if output_path.exists():
            skipped_count += 1
            continue

        try:
            with urlopen(selected_object['url'], timeout=30) as response:
                image = Image.open(BytesIO(response.read())).convert('RGB')

            left = max(0, selected_object['x'])
            top = max(0, selected_object['y'])
            right = min(image.width, selected_object['x'] + selected_object['w'])
            bottom = min(image.height, selected_object['y'] + selected_object['h'])

            if right <= left or bottom <= top:
                failed_downloads.append(selected_object)
                continue

            crop = image.crop((left, top, right, bottom))
            crop.save(output_path, quality=95)

            downloaded_count += 1

        except Exception:
            failed_downloads.append(selected_object)

    with open(failed_path, 'w', encoding='utf-8') as file:
        json.dump(failed_downloads, file, indent=2)

    print('Downloaded crops:', downloaded_count)
    print('Skipped existing crops:', skipped_count)
    print('Failed downloads:', len(failed_downloads))