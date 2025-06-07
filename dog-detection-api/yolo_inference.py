import os
import torch
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error
import tempfile
import json
from datetime import datetime
import pandas as pd


class YOLODogInference:
    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5001",
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        minio_bucket: str = "mlflow-artifacts",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_bucket = minio_bucket
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
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
        
       
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("dog_detection_yolov8_v2")
        
        print(f"🔍 YOLODogInference ініціалізовано:")
        print(f"   - MLFlow URI: {mlflow_tracking_uri}")
        print(f"   - MinIO: {minio_endpoint}")
        print(f"   - Bucket: {minio_bucket}")
        print(f"   - Confidence threshold: {confidence_threshold}")
        print(f"   - IoU threshold: {iou_threshold}")
        print(f"   - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    def list_available_models(self):
        """Показує доступні моделі в MinIO"""
        try:
            models = []
            objects = self.minio_client.list_objects(self.minio_bucket, prefix="models/", recursive=True)
            
            for obj in objects:
                if obj.object_name.endswith('.pt'):
                    models.append(obj.object_name)
            
            print(f"\n📋 Доступні моделі в MinIO:")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model}")
            
            return models
        except Exception as e:
            print(f"❌ Помилка отримання списку моделей: {e}")
            return []
    
    def download_model_from_minio(self, model_path: str, local_path: str = None):
        """Завантажує модель з MinIO"""
        try:
            if local_path is None:
                local_path = f"temp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            
            self.minio_client.fget_object(
                self.minio_bucket,
                model_path,
                local_path
            )
            print(f"✅ Модель завантажена: {model_path} -> {local_path}")
            return local_path
        except Exception as e:
            print(f"❌ Помилка завантаження моделі: {e}")
            return None
    
    def get_latest_model(self):
        try:
            experiment = mlflow.get_experiment_by_name("dog_detection_yolov8_v2")
            if experiment is not None:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=5
                )
                
                if not runs.empty:
                    for _, run in runs.iterrows():
                        for param_name in ['params.best_model_minio_path', 'params.last_model_minio_path']:
                            if param_name in run and pd.notna(run[param_name]):
                                minio_path = run[param_name]
                                if minio_path.startswith('s3://'):
                                    best_model_path = minio_path.replace(f's3://{self.minio_bucket}/', '')
                                    print(f"🎯 Знайдена модель через MLFlow: {best_model_path}")
                                    return best_model_path
            
            print("🔍 MLFlow не знайшов модель, шукаємо найновішу в MinIO...")
            models = []
            objects = self.minio_client.list_objects(self.minio_bucket, prefix="models/", recursive=True)
            
            for obj in objects:
                if obj.object_name.endswith('best.pt'):
                    models.append((obj.object_name, obj.last_modified))
            
            if models:
                models.sort(key=lambda x: x[1], reverse=True)
                latest_model = models[0][0]
                print(f"🎯 Знайдена найновіша модель в MinIO: {latest_model}")
                return latest_model
            else:
                print("❌ Не знайдено жодної моделі best.pt в MinIO")
                return None
                
        except Exception as e:
            print(f"❌ Помилка отримання останньої моделі: {e}")
            return None
    
    def load_model(self, model_path: str = None):
        try:
            if model_path is None:
                model_path = self.get_latest_model()
                if model_path is None:
                    print("❌ Не вдалося знайти натреновану модель")
                    return None
            
            
            if not os.path.exists(model_path):
                local_model_path = self.download_model_from_minio(model_path)
                if local_model_path is None:
                    return None
                model_path = local_model_path
            
            model = YOLO(model_path)
            print(f"✅ Модель завантажена: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Помилка завантаження моделі: {e}")
            return None
    
    def prepare_test_images(self, test_dir: str = "test_images"):
        """Підготовка тестових зображень"""
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"❌ Папка {test_dir} не знайдена")
            return []
        
        supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = set()
        
        for format_pattern in supported_formats:
            image_files.update(test_path.glob(format_pattern))
            image_files.update(test_path.glob(format_pattern.upper()))
        
        image_files = sorted(list(image_files))
        
        print(f"📸 Знайдено {len(image_files)} тестових зображень в {test_dir}")
        for i, img in enumerate(image_files, 1):
            print(f"   {i}. {img.name}")
        
        return image_files
    
    def run_inference_on_image(self, model, image_path: Path, save_results: bool = True):
        """Виконує інференс на одному зображенні"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Не вдалося завантажити зображення: {image_path}")
                return None
            
            results = model(
                str(image_path),
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': 'dog'
                    })
            
            if save_results and detections:
                output_dir = Path("inference_results")
                output_dir.mkdir(exist_ok=True)
                
                annotated_image = image.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    label = f"dog: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                output_path = output_dir / f"result_{image_path.name}"
                cv2.imwrite(str(output_path), annotated_image)
                
                print(f"  💾 Результат збережено: {output_path}")
            
            return {
                'image_path': str(image_path),
                'detections': detections,
                'detection_count': len(detections)
            }
            
        except Exception as e:
            print(f"❌ Помилка інференсу для {image_path}: {e}")
            return None
    
    def run_batch_inference(self, model_path: str = None, test_dir: str = "test_images", save_results: bool = True):
        """Виконує інференс на всіх зображеннях з тестової папки"""
        
        model = self.load_model(model_path)
        if model is None:
            return None
        
        image_files = self.prepare_test_images(test_dir)
        if not image_files:
            return None
        
        if save_results:
            results_dir = Path("inference_results")
            results_dir.mkdir(exist_ok=True)
            print(f"📁 Результати будуть збережені в: {results_dir}")
        
        all_results = []
        total_detections = 0
        
        print(f"\n🚀 Початок інференсу...")
        print("-" * 50)
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Обробка: {image_path.name}")
            
            result = self.run_inference_on_image(model, image_path, save_results)
            if result:
                all_results.append(result)
                detection_count = result['detection_count']
                total_detections += detection_count
                
                if detection_count > 0:
                    print(f"  🐕 Знайдено собак: {detection_count}")
                    for j, det in enumerate(result['detections'], 1):
                        print(f"    {j}. Confidence: {det['confidence']:.3f}")
                else:
                    print(f"  ❌ Собак не знайдено")
            else:
                print(f"  ❌ Помилка обробки")
        
        if save_results and all_results:
            results_file = Path("inference_results") / "inference_summary.json"
            summary = {
                'total_images': len(image_files),
                'processed_images': len(all_results),
                'total_detections': total_detections,
                'images_with_dogs': len([r for r in all_results if r['detection_count'] > 0]),
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n📊 Збережено звіт: {results_file}")
        
        print("\n" + "=" * 50)
        print("📈 ПІДСУМКИ ІНФЕРЕНСУ")
        print("=" * 50)
        print(f"📸 Оброблено зображень: {len(all_results)}/{len(image_files)}")
        print(f"🐕 Всього знайдено собак: {total_detections}")
        print(f"📷 Зображень з собаками: {len([r for r in all_results if r['detection_count'] > 0])}")
        print(f"🎯 Точність (Confidence): {self.confidence_threshold}")
        print(f"🔧 IoU threshold: {self.iou_threshold}")
        
        if save_results:
            print(f"💾 Результати збережені в: inference_results/")
        
        return all_results
    
    def test_single_image(self, image_path: str, model_path: str = None):
        model = self.load_model(model_path)
        if model is None:
            return None
        
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"❌ Зображення не знайдено: {image_path}")
            return None
        
        print(f"🔍 Тестування зображення: {image_path}")
        result = self.run_inference_on_image(model, image_file, save_results=True)
        
        if result and result['detection_count'] > 0:
            print(f"✅ Знайдено {result['detection_count']} собак")
            for i, det in enumerate(result['detections'], 1):
                print(f"  {i}. Confidence: {det['confidence']:.3f}")
        else:
            print("❌ Собак не знайдено")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Dog Detection Inference')
    parser.add_argument('--model', type=str, help='Шлях до моделі (якщо не вказано, використовується остання)')
    parser.add_argument('--test_dir', type=str, default='test_images', help='Папка з тестовими зображеннями')
    parser.add_argument('--single_image', type=str, help='Тестувати одне зображення')
    parser.add_argument('--confidence', type=float, default=0.5, help='Поріг впевненості')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU поріг')
    parser.add_argument('--no_save', action='store_true', help='Не зберігати результати')
    parser.add_argument('--list_models', action='store_true', help='Показати доступні моделі')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5001', help='MLFlow URI')
    parser.add_argument('--minio_endpoint', type=str, default='localhost:9000', help='MinIO endpoint')
    parser.add_argument('--minio_bucket', type=str, default='mlflow-artifacts', help='MinIO bucket')
    
    args = parser.parse_args()
    
    print("🔍 YOLOv8 Dog Detection Inference")
    print("=" * 50)
    
    inference = YOLODogInference(
        mlflow_tracking_uri=args.mlflow_uri,
        minio_endpoint=args.minio_endpoint,
        minio_bucket=args.minio_bucket,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    try:
        if args.list_models:
            inference.list_available_models()
        
        elif args.single_image:
            inference.test_single_image(args.single_image, args.model)
        
        else:
            save_results = not args.no_save
            results = inference.run_batch_inference(
                model_path=args.model,
                test_dir=args.test_dir,
                save_results=save_results
            )
            
            if results:
                print("✅ Інференс успішно завершено!")
            else:
                print("❌ Інференс не завершено")
    
    except Exception as e:
        print(f"❌ Помилка під час інференсу: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()