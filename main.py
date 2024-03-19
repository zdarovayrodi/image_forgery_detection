import os
import csv
from PIL import Image


def get_image_info(image_path):
    image = Image.open(image_path)
    info = {
        'Mode': image.mode,
        'Width': image.width,
        'Height': image.height,
        **image.info,
    }
    return info


if __name__ == "__main__":
    original_folder = 'img/original/'
    generated_folder = 'img/generated/'
    csv_file_path = 'image_properties.csv'

    all_info_keys = set()

    for original_filename in os.listdir(original_folder):
        if original_filename.endswith('.jpg') or original_filename.endswith('.png'):
            original_image_path = os.path.join(original_folder, original_filename)
            generated_image_path = os.path.join(generated_folder, original_filename)

            if not os.path.exists(generated_image_path):
                for generated_filename in os.listdir(generated_folder):
                    original_filename_no_ext = original_filename.split('.')[0]
                    if generated_filename.startswith(original_filename_no_ext):
                        generated_image_path = os.path.join(generated_folder, generated_filename)
                        break

            if os.path.exists(generated_image_path):
                original_info = get_image_info(original_image_path)
                generated_info = get_image_info(generated_image_path)

                all_info_keys.update(original_info.keys())
                all_info_keys.update(generated_info.keys())

    with open(csv_file_path, 'w', newline='') as csv_file:
        fieldnames = ['Filename', 'Width', 'Height', 'Mode', *all_info_keys]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()

        for original_filename in os.listdir(original_folder):
            if original_filename.endswith('.jpg') or original_filename.endswith('.png'):
                original_image_path = os.path.join(original_folder, original_filename)
                generated_image_path = os.path.join(generated_folder, original_filename)

                if not os.path.exists(generated_image_path):
                    for generated_filename in os.listdir(generated_folder):
                        original_filename_no_ext = original_filename.split('.')[0]
                        if generated_filename.startswith(original_filename_no_ext):
                            generated_image_path = os.path.join(generated_folder, generated_filename)
                            break

                generated_filename = os.path.basename(generated_image_path)
                
                if os.path.exists(generated_image_path):
                    original_info = get_image_info(original_image_path)
                    generated_info = get_image_info(generated_image_path)

                    writer.writerow({**{'Filename': original_filename}, **original_info})
                    writer.writerow({**{'Filename': generated_filename}, **generated_info})
