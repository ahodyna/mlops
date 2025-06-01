import re
import shutil
from pathlib import Path


def extract_number_from_filename(filename):
    """Витягує номер з кінця назви файлу"""
    # Для анотацій: "0ed06d6a_13.txt" -> 13
    # Для зображень: "13.png" -> 13
    
    stem = Path(filename).stem
    
    match = re.search(r'_(\d+)$', stem) 
    if match:
        return int(match.group(1))
    
    if stem.isdigit():
        return int(stem)
    
    match = re.search(r'(\d+)$', stem)
    if match:
        return int(match.group(1))
    
    return None


def create_number_based_mapping():
    
    print(" Створення mapping на основі номерів...")
    
    image_dir = Path('dataset/images')
    label_dir = Path('dataset/labels')
    
    image_files = [f for f in image_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    label_files = [f for f in label_dir.glob('*.txt')]
    
    print(f"📊 Знайдено:")
    print(f"   - Зображення: {len(image_files)}")
    print(f"   - Анотації: {len(label_files)}")
    
    # Створюємо словники номер -> файл
    image_map = {}  # номер -> файл зображення
    label_map = {}  # номер -> файл анотації
    
    print(f"\n🔍 Аналіз номерів в зображеннях:")
    for img_file in image_files:
        number = extract_number_from_filename(img_file.name)
        if number is not None:
            image_map[number] = img_file
            print(f"   {img_file.name} → номер {number}")
        else:
            print(f"    Не вдалося витягти номер з {img_file.name}")
    
    print(f"\n Аналіз номерів в анотаціях:")
    for lbl_file in label_files:
        number = extract_number_from_filename(lbl_file.name)
        if number is not None:
            label_map[number] = lbl_file
            print(f"   {lbl_file.name} → номер {number}")
        else:
            print(f"    Не вдалося витягти номер з {lbl_file.name}")
    
    common_numbers = set(image_map.keys()) & set(label_map.keys())
    
    print(f"\n Результати mapping:")
    print(f"   - Унікальних номерів зображень: {len(image_map)}")
    print(f"   - Унікальних номерів анотацій: {len(label_map)}")
    print(f"   - Співпадінь: {len(common_numbers)}")
    
    if len(common_numbers) > 0:
        print(f"\n Знайдені пари:")
        
        output_dir = Path('dataset_numbered')
        output_img_dir = output_dir / 'images'
        output_lbl_dir = output_dir / 'labels'
        
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        output_img_dir.mkdir(parents=True)
        output_lbl_dir.mkdir(parents=True)
        
        successful_pairs = 0
        
        for number in sorted(common_numbers):
            img_file = image_map[number]
            lbl_file = label_map[number]
            
            new_img_name = f"dog_{number:04d}{img_file.suffix}"
            new_lbl_name = f"dog_{number:04d}.txt"
            
            try:
               
                shutil.copy2(img_file, output_img_dir / new_img_name)
                shutil.copy2(lbl_file, output_lbl_dir / new_lbl_name)
                
                print(f"   {number:2d}: {img_file.name} + {lbl_file.name} → dog_{number:04d}")
                successful_pairs += 1
                
            except Exception as e:
                print(f"   ❌ Помилка з номером {number}: {e}")
        
        # Створюємо dataset.yaml
        yaml_content = f"""# Dataset for dog detection training
path: dataset_numbered
train: images
val: images
nc: 1
names:
  0: dog
"""
        
        with open('dataset_numbered.yaml', 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"\n Mapping завершено!")
        print(f"   - Успішних пар: {successful_pairs}")
        print(f"   - Структура створена: dataset_numbered/")
        print(f"   - Конфігурація: dataset_numbered.yaml")
    
        missing_images = set(label_map.keys()) - set(image_map.keys())
        missing_labels = set(image_map.keys()) - set(label_map.keys())
        
        if missing_images:
            print(f"\n Анотації без зображень (номери): {sorted(list(missing_images)[:10])}")
        if missing_labels:
            print(f"Зображення без анотацій (номери): {sorted(list(missing_labels)[:10])}")
        
        if successful_pairs > 0:
            print(f"\n Готово до тренування:")
            print(f"   python src/train_yolo.py --epochs 5 --batch 4")
            print(f"   (використовуйте dataset_numbered.yaml)")
        
        return successful_pairs > 0
    
    else:
        print(f"\n Співпадіння не знайдені")
        print(f" Перевірте що номери в назвах файлів співпадають")
        return False


if __name__ == "__main__":
    create_number_based_mapping()