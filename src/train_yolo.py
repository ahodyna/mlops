import os
import torch
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from pathlib import Path
import argparse
import yaml
from datetime import datetime
import shutil
import tempfile
from minio import Minio
from minio.error import S3Error


class YOLODogTrainer:

    def __init__(
        self,
        model_name: str = 'yolov8s.pt',  # Використовуємо більшу модель
        data_config: str = 'dataset_numbered.yaml',
        mlflow_tracking_uri: str = "http://localhost:5001",
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        minio_bucket: str = "mlflow-artifacts"
    ):
        self.model_name = model_name
        self.data_config = data_config
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_bucket = minio_bucket
        
        os.environ['AWS_ACCESS_KEY_ID'] = minio_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = minio_secret_key
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://{minio_endpoint}'
        
        self.minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        self.setup_minio_bucket()
        
        # MLFlow налаштування
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("dog_detection_yolov8_small_dataset")
        
        print(f"🐕 YOLODogTrainer ініціалізовано (для малого датасету):")
        print(f"   - Модель: {model_name}")
        print(f"   - Dataset config: {data_config}")
        print(f"   - MLFlow URI: {mlflow_tracking_uri}")
        print(f"   - MinIO: {minio_endpoint}")
        print(f"   - Bucket: {minio_bucket}")
        print(f"   - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    def setup_minio_bucket(self):
        """Створює bucket в MinIO якщо не існує"""
        try:
            if not self.minio_client.bucket_exists(self.minio_bucket):
                self.minio_client.make_bucket(self.minio_bucket)
                print(f"✅ MinIO bucket '{self.minio_bucket}' створено")
            else:
                print(f"✅ MinIO bucket '{self.minio_bucket}' готовий")
        except Exception as e:
            print(f"❌ Помилка MinIO: {e}")
    
    def upload_to_minio(self, local_path: Path, object_name: str) -> bool:
        """Завантажує файл в MinIO"""
        try:
            self.minio_client.fput_object(
                self.minio_bucket,
                object_name,
                str(local_path)
            )
            print(f"📤 Завантажено в MinIO: {object_name}")
            return True
        except Exception as e:
            print(f"❌ Помилка завантаження в MinIO: {e}")
            return False
    
    def log_metrics_to_mlflow(self, run_dir: Path):
        """Логування метрик тренування у MLFlow"""
        try:
            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                
                # Логування метрик по епохах
                for idx, row in df.iterrows():
                    epoch = idx + 1
                    
                    # Основні метрики
                    if 'metrics/mAP50(B)' in row:
                        mlflow.log_metric("mAP50", float(row['metrics/mAP50(B)']), step=epoch)
                    if 'metrics/mAP50-95(B)' in row:
                        mlflow.log_metric("mAP50-95", float(row['metrics/mAP50-95(B)']), step=epoch)
                    if 'metrics/precision(B)' in row:
                        mlflow.log_metric("precision", float(row['metrics/precision(B)']), step=epoch)
                    if 'metrics/recall(B)' in row:
                        mlflow.log_metric("recall", float(row['metrics/recall(B)']), step=epoch)
                    
                    # Loss метрики
                    if 'train/box_loss' in row:
                        mlflow.log_metric("train_box_loss", float(row['train/box_loss']), step=epoch)
                    if 'train/cls_loss' in row:
                        mlflow.log_metric("train_cls_loss", float(row['train/cls_loss']), step=epoch)
                    if 'train/dfl_loss' in row:
                        mlflow.log_metric("train_dfl_loss", float(row['train/dfl_loss']), step=epoch)
                    if 'val/box_loss' in row:
                        mlflow.log_metric("val_box_loss", float(row['val/box_loss']), step=epoch)
                    if 'val/cls_loss' in row:
                        mlflow.log_metric("val_cls_loss", float(row['val/cls_loss']), step=epoch)
                    if 'val/dfl_loss' in row:
                        mlflow.log_metric("val_dfl_loss", float(row['val/dfl_loss']), step=epoch)
                
                final_row = df.iloc[-1]
                mlflow.log_metrics({
                    "final_mAP50": float(final_row.get('metrics/mAP50(B)', 0)),
                    "final_mAP50-95": float(final_row.get('metrics/mAP50-95(B)', 0)),
                    "final_precision": float(final_row.get('metrics/precision(B)', 0)),
                    "final_recall": float(final_row.get('metrics/recall(B)', 0)),
                    "final_train_loss": float(final_row.get('train/box_loss', 0)),
                    "final_val_loss": float(final_row.get('val/box_loss', 0))
                })
                
                print(f"📊 Метрики збережені у MLFlow:")
                print(f"   - mAP50: {final_row.get('metrics/mAP50(B)', 0):.3f}")
                print(f"   - Precision: {final_row.get('metrics/precision(B)', 0):.3f}")
                print(f"   - Recall: {final_row.get('metrics/recall(B)', 0):.3f}")
                
        except Exception as e:
            print(f"⚠️ Помилка логування метрик: {e}")
    
    def prepare_dataset(self):
        """Перевіряє та підготовує dataset для YOLOv8"""
        
        required_paths = [
            Path('dataset_numbered/images'),
            Path('dataset_numbered/labels')
        ]
        
        for path in required_paths:
            if not path.exists():
                print(f"❌ Папка не знайдена: {path}")
                print(f"Спочатку запустіть: python create_number_mapping.py")
                return False
        
        image_files = list(Path('dataset_numbered/images').glob('*.jpg')) + \
                     list(Path('dataset_numbered/images').glob('*.png')) + \
                     list(Path('dataset_numbered/images').glob('*.jpeg'))
        
        label_files = list(Path('dataset_numbered/labels').glob('*.txt'))
        
        print(f"📁 Знайдено файлів:")
        print(f"   - Зображення: {len(image_files)}")
        print(f"   - Анотації: {len(label_files)}")
        
        if len(image_files) == 0:
            print("❌ Не знайдено зображень в dataset_numbered/images/")
            return False
        
        if len(label_files) == 0:
            print("❌ Не знайдено анотацій в dataset_numbered/labels/")
            return False
        
        # Перевірка відповідності файлів
        matched_files = 0
        for img_file in image_files:
            label_file = Path('dataset_numbered/labels') / f"{img_file.stem}.txt"
            if label_file.exists():
                matched_files += 1
        
        print(f"   - Пари зображення-анотація: {matched_files}")
        
        if matched_files == 0:
            print("❌ Жодне зображення не має відповідної анотації")
            return False
    
        
        print("✅ Dataset готовий для тренування")
        return True
    
    def create_train_val_split(self, train_ratio: float = 0.9): 
        """
        Створює train/val розподіл для YOLOv8 (оптимізовано для малого датасету)
        """
        
        train_img_dir = Path('dataset_numbered/train/images')
        train_lbl_dir = Path('dataset_numbered/train/labels')
        val_img_dir = Path('dataset_numbered/val/images')
        val_lbl_dir = Path('dataset_numbered/val/labels')
        
        for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend(Path('dataset_numbered/images').glob(ext))
        
        valid_pairs = []
        for img_file in image_files:
            label_file = Path('dataset_numbered/labels') / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
        
        import random
        random.shuffle(valid_pairs)
        
        train_count = int(len(valid_pairs) * train_ratio)
        train_pairs = valid_pairs[:train_count]
        val_pairs = valid_pairs[train_count:]
        
        print(f"📊 Створення train/val розподілу (оптимізовано для малого датасету):")
        print(f"   - Train: {len(train_pairs)} файлів ({train_ratio*100:.0f}%)")
        print(f"   - Val: {len(val_pairs)} файлів ({(1-train_ratio)*100:.0f}%)")
        
        for img_file, label_file in train_pairs:
            shutil.copy2(img_file, train_img_dir / img_file.name)
            shutil.copy2(label_file, train_lbl_dir / label_file.name)
        
        for img_file, label_file in val_pairs:
            shutil.copy2(img_file, val_img_dir / img_file.name)
            shutil.copy2(label_file, val_lbl_dir / label_file.name)
        
        updated_config = {
            'path': 'dataset_numbered',
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,
            'names': {0: 'dog'}
        }
        
        with open('dataset_numbered_split.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(updated_config, f)
        
        print("✅ Train/val розподіл створено")
        return 'dataset_numbered_split.yaml'
    
    def train(
        self,
        epochs: int = 50,
        imgsz: int = 416, 
        batch: int = 2, 
        lr0: float = 0.0001,
        project: str = "runs/detect",
        name: str = "dog_detection_small_dataset"
    ):
        """
        Тренування YOLOv8 оптимізоване для малого датасету
        """
        
        if not self.prepare_dataset():
            print("❌ Помилка підготовки dataset")
            return None
        
        data_config_path = self.create_train_val_split()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_project = Path(temp_dir) / "runs" / "detect"
            
            model = YOLO(self.model_name)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with mlflow.start_run() as run:
                run_id = run.info.run_id[:8]
                model_prefix = f"models/dog_detection_small_{timestamp}_{run_id}"
                
                # Параметри для малого датасету
                params = {
                    "model": self.model_name,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "lr0": lr0,
                    "optimizer": "AdamW",  # Кращий для малих датасетів
                    "data_config": data_config_path,
                    "minio_bucket": self.minio_bucket,
                    "dataset_optimization": "small_dataset",
                    "train_ratio": 0.9,
                    "augmentation": "aggressive"
                }
                
                mlflow.log_params(params)
                
                print(f"\n🚀 Початок тренування YOLOv8 для малого датасету...")
                print(f"📂 Тимчасова папка: {temp_project}")
                print(f"🔍 MLFlow Run ID: {run_id}")
                print(f"⚙️  Параметри оптимізовані для малого датасету:")
                print(f"   - Епохи: {epochs}")
                print(f"   - Batch size: {batch}")
                print(f"   - Learning rate: {lr0}")
                print(f"   - Image size: {imgsz}")
                print("-" * 60)
                
                # Тренування з агресивною аугментацією
                results = model.train(
                    data=data_config_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    lr0=lr0,
                    optimizer='AdamW',
                    # Агресивна аугментація для малого датасету
                    hsv_h=0.015,        # Зміна відтінку
                    hsv_s=0.7,          # Зміна насиченості  
                    hsv_v=0.4,          # Зміна яскравості
                    degrees=20.0,       # Поворот
                    translate=0.2,      # Зсув
                    scale=0.8,          # Масштабування
                    shear=0.15,         # Зсув
                    perspective=0.0005, # Перспектива
                    flipud=0.5,         # Вертикальний переворот
                    fliplr=0.5,         # Горизонтальний переворот
                    mosaic=1.0,         # Mosaic аугментація
                    mixup=0.15,         # MixUp аугментація
                    copy_paste=0.3,     # Copy-paste аугментація
                    # Regularization
                    dropout=0.2,        # Dropout
                    weight_decay=0.0005, # Weight decay
                    # Налаштування навчання
                    patience=50,        # Early stopping
                    save_period=20,     # Збереження кожні 20 епох
                    project=str(temp_project),
                    name=name,
                    save=True,
                    plots=True,
                    verbose=True
                )
                
                run_dir = temp_project / name
                
                print(f"\n📊 Логування метрик у MLFlow...")
                self.log_metrics_to_mlflow(run_dir)
                
                best_model_path = run_dir / "weights" / "best.pt"
                last_model_path = run_dir / "weights" / "last.pt"
                
                print(f"\n📤 Завантаження моделей в MinIO...")
                
                if best_model_path.exists():
                    best_object_name = f"{model_prefix}/best.pt"
                    if self.upload_to_minio(best_model_path, best_object_name):
                        mlflow.log_param("best_model_minio_path", f"s3://{self.minio_bucket}/{best_object_name}")
                        print(f"✅ Найкраща модель: s3://{self.minio_bucket}/{best_object_name}")
                
                if last_model_path.exists():
                    last_object_name = f"{model_prefix}/last.pt"
                    if self.upload_to_minio(last_model_path, last_object_name):
                        mlflow.log_param("last_model_minio_path", f"s3://{self.minio_bucket}/{last_object_name}")
                        print(f"📦 Остання модель: s3://{self.minio_bucket}/{last_object_name}")
                
                # Збереження графіків та результатів
                results_dir = run_dir
                if results_dir.exists():
                    for plot_file in results_dir.glob("*.png"):
                        mlflow.log_artifact(str(plot_file), "plots")
                        plot_object_name = f"{model_prefix}/plots/{plot_file.name}"
                        self.upload_to_minio(plot_file, plot_object_name)
                   
                    results_csv = results_dir / "results.csv"
                    if results_csv.exists():
                        mlflow.log_artifact(str(results_csv), "training_results")
                        csv_object_name = f"{model_prefix}/results.csv"
                        self.upload_to_minio(results_csv, csv_object_name)
                
                print(f"\n🎉 Тренування завершено!")
                print(f"📊 Результати та моделі збережені в MinIO bucket: {self.minio_bucket}")
                print(f"🌐 MinIO Console: http://localhost:9009")
                print(f"📈 MLFlow UI: http://localhost:5001")
                print(f"🗂️ Модель prefix: {model_prefix}")
                
                return model, results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Dog Detection (Small Dataset Optimized)')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='YOLOv8 model (рекомендується s або m для малих датасетів)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (більше для малого датасету)')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size (менше для кращої генералізації)')
    parser.add_argument('--batch', type=int, default=2, help='Batch size (малий для малого датасету)')
    parser.add_argument('--lr0', type=float, default=0.0001, help='Learning rate (менший для малого датасету)')
    parser.add_argument('--name', type=str, default='dog_detection_small', help='Run name')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5001', help='MLFlow URI')
    parser.add_argument('--minio_endpoint', type=str, default='localhost:9000', help='MinIO endpoint')
    parser.add_argument('--minio_bucket', type=str, default='mlflow-artifacts', help='MinIO bucket')
    
    args = parser.parse_args()
    
    print("🐕 YOLOv8 Dog Detection Training - Optimized for Small Datasets")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image Size: {args.imgsz}")
    print(f"Batch Size: {args.batch}")
    print(f"Learning Rate: {args.lr0}")
    print(f"MLFlow URI: {args.mlflow_uri}")
    print(f"MinIO Endpoint: {args.minio_endpoint}")
    print(f"MinIO Bucket: {args.minio_bucket}")
    print("=" * 70)
    
    trainer = YOLODogTrainer(
        model_name=args.model,
        mlflow_tracking_uri=args.mlflow_uri,
        minio_endpoint=args.minio_endpoint,
        minio_bucket=args.minio_bucket
    )
    
    try:
        result = trainer.train(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            name=args.name
        )
        
        if result is not None:
            print("✅ Тренування успішно завершено!")
            print("📦 Всі артефакти збережені в MinIO!")
        else:
            print("❌ Тренування не завершено")
        
    except Exception as e:
        print(f"❌ Помилка під час тренування: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()