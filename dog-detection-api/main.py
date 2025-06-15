from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime
import asyncio
import uvicorn
import time
import cv2
import numpy as np
import pandas as pd

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Evidently для drift detection
try:
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️ Evidently не встановлено. Drift detection недоступний.")

from yolo_inference import YOLODogInference

app = FastAPI(
    title="🐕 Dog Detection API",
    description="YOLOv8 Dog Detection Service with MLFlow and MinIO integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictions_total = Counter('dog_predictions_total', 'Total predictions made', ['status'])

# Час обробки
processing_time = Histogram(
    'dog_processing_seconds', 
    'Time spent processing images',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

dogs_detected = Counter('dogs_detected_total', 'Total dogs detected')

class SimpleDriftMonitor:
    def __init__(self):
        self.reference_data = []
        self.current_batch = []
        self.batch_size = 20
        
    def extract_features(self, image_data, processing_time, dogs_count):
        """Витягує прості ознаки з зображення"""
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            return {
                'width': img.shape[1],
                'height': img.shape[0],
                'brightness': float(np.mean(img)),
                'processing_time': processing_time,
                'dogs_count': dogs_count
            }
        return None
    
    def add_sample(self, image_data, processing_time, dogs_count):
        """Додає зразок для аналізу"""
        features = self.extract_features(image_data, processing_time, dogs_count)
        if features:
            self.current_batch.append(features)
            
            if len(self.current_batch) >= self.batch_size:
                self.check_drift()
                self.current_batch = []
    
    def set_reference_data(self):
        if self.current_batch:
            self.reference_data = self.current_batch.copy()
            print(f"✅ Встановлено {len(self.reference_data)} еталонних зразків")
    
    def check_drift(self):   
        if not EVIDENTLY_AVAILABLE or not self.reference_data:
            return
            
        try:
            ref_df = pd.DataFrame(self.reference_data)
            curr_df = pd.DataFrame(self.current_batch)
            
            report = Report(metrics=[DatasetDriftMetric()])
            report.run(reference_data=ref_df, current_data=curr_df)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"drift_report_{timestamp}.html"
            report.save_html(report_path)
            
            results = report.as_dict()
            drift_detected = results['metrics'][0]['result']['dataset_drift']
            
            if drift_detected:
                print(f"🚨 ВИЯВЛЕНО ДРІФТ! Звіт збережено: {report_path}")
            else:
                print(f"✅ Дріфт не виявлено. Звіт: {report_path}")
                
        except Exception as e:
            print(f"❌ Помилка перевірки дріфту: {e}")

# Global services - оголошуємо тут, до використання!
inference_service = None
drift_monitor = SimpleDriftMonitor()

@app.on_event("startup")
async def startup_event():
    """Ініціалізація сервісу при запуску - test"""
    global inference_service  # ← Тепер global ПЕРЕД використанням
    
    # Перевіряємо чи це CI/CD середовище
    is_ci = os.getenv("CI", "false").lower() == "true"
    is_github_actions = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
    
    if is_ci or is_github_actions:
        print("🧪 CI/CD режим - пропускаємо ініціалізацію MLflow")
        inference_service = None
        return
    
    try:
        mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        minio_bucket = os.getenv("MINIO_BUCKET", "mlflow-artifacts")
        
        print(f"🔄 Підключення до MLflow: {mlflow_uri}")
        print(f"🔄 Підключення до MinIO: {minio_endpoint}")
        
        inference_service = YOLODogInference(
            mlflow_tracking_uri=mlflow_uri,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            confidence_threshold=0.5,
            iou_threshold=0.45
        )
        print("✅ Dog Detection API запущено успішно!")
    except Exception as e:
        print(f"❌ Помилка ініціалізації: {e}")
        print("💡 API працює в обмеженому режимі")
        inference_service = None

@app.get("/")
async def root():
    """Головна сторінка API"""
    is_ci = os.getenv("CI", "false").lower() == "true"
    is_github_actions = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
    
    return {
        "message": "🐕 Dog Detection API",
        "version": "1.0.0",
        "status": "running",
        "mode": "ci-test" if (is_ci or is_github_actions) else "production",
        "inference_available": inference_service is not None,
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "metrics": "/metrics",
            "drift_baseline": "/set-drift-baseline"
        }
    }

@app.get("/health")
async def health_check():
    """Перевірка стану сервісу"""
    is_ci = os.getenv("CI", "false").lower() == "true"
    is_github_actions = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "dog-detection-api",
        "mode": "ci-test" if (is_ci or is_github_actions) else "production",
        "inference_available": inference_service is not None
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus метрики"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/set-drift-baseline")
async def set_drift_baseline():
    """Встановлює поточні дані як еталонні для drift detection"""
    drift_monitor.set_reference_data()
    return {"message": "Еталонні дані встановлені", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models():
    """Отримати список доступних моделей"""
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference сервіс недоступний (CI режим або помилка ініціалізації)")
    
    try:
        models = inference_service.list_available_models()
        latest_model = inference_service.get_latest_model()
        
        return {
            "available_models": models,
            "latest_model": latest_model,
            "total_count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка отримання моделей: {str(e)}")

@app.post("/predict")
async def predict_single_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold"),
    model_path: Optional[str] = Query(None, description="Specific model path")
):
    """Інференс на одному зображенні"""
    
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference сервіс недоступний (CI режим або помилка ініціалізації)")
    
    start_time = time.time()
    
    if not file.content_type.startswith('image/'):
        predictions_total.labels(status='error').inc()
        raise HTTPException(status_code=400, detail="Файл повинен бути зображенням")
    
    temp_path = None
    try:
        image_data = await file.read()
        file.file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        inference_service.confidence_threshold = confidence
        inference_service.iou_threshold = iou
        
        model = inference_service.load_model(model_path)
        if model is None:
            predictions_total.labels(status='error').inc()
            raise HTTPException(status_code=500, detail="Не вдалося завантажити модель")
        
        result = inference_service.run_inference_on_image(
            model, 
            Path(temp_path), 
            save_results=False
        )
        
        if result is None:
            predictions_total.labels(status='error').inc()
            raise HTTPException(status_code=500, detail="Помилка обробки зображення")
        
        process_time = time.time() - start_time
        processing_time.observe(process_time)
        predictions_total.labels(status='success').inc()
        dogs_detected.inc(result['detection_count'])
        
        # Додаємо до drift monitoring
        drift_monitor.add_sample(image_data, process_time, result['detection_count'])
        
        return {
            "filename": file.filename,
            "detections": result['detections'],
            "detection_count": result['detection_count'],
            "processing_time": process_time,
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        predictions_total.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=f"Помилка інференсу: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.post("/batch-predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    model_path: Optional[str] = Query(None)
):
    """Batch інференс на декількох зображеннях"""
    
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference сервіс недоступний (CI режим або помилка ініціалізації)")
    
    if len(files) > 10:
        predictions_total.labels(status='error').inc()
        raise HTTPException(status_code=400, detail="Максимум 10 файлів за раз")
    
    for file in files:
        if not file.content_type.startswith('image/'):
            predictions_total.labels(status='error').inc()
            raise HTTPException(status_code=400, detail=f"Файл {file.filename} не є зображенням")
    
    start_time = time.time()
    temp_files = []
    
    try:
        inference_service.confidence_threshold = confidence
        inference_service.iou_threshold = iou
        
        model = inference_service.load_model(model_path)
        if model is None:
            predictions_total.labels(status='error').inc()
            raise HTTPException(status_code=500, detail="Не вдалося завантажити модель")
        
        results = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            result = inference_service.run_inference_on_image(
                model, 
                Path(temp_path), 
                save_results=False
            )
            
            if result:
                results.append({
                    "filename": file.filename,
                    "detections": result['detections'],
                    "detection_count": result['detection_count']
                })
                dogs_detected.inc(result['detection_count'])
        
        process_time = time.time() - start_time
        processing_time.observe(process_time)
        predictions_total.labels(status='success').inc()
        
        total_detections = sum(r['detection_count'] for r in results)
        images_with_dogs = len([r for r in results if r['detection_count'] > 0])
        
        return {
            "results": results,
            "summary": {
                "total_images": len(files),
                "processed_images": len(results),
                "total_detections": total_detections,
                "images_with_dogs": images_with_dogs,
                "processing_time": process_time,
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        predictions_total.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=f"Помилка batch інференсу: {str(e)}")
    finally:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

@app.get("/stats")
async def get_stats():
    """Статистика сервісу"""
    return {
        "service": "dog-detection-api",
        "model_info": {
            "latest_model": inference_service.get_latest_model() if inference_service else None,
            "confidence_threshold": inference_service.confidence_threshold if inference_service else None,
            "iou_threshold": inference_service.iou_threshold if inference_service else None,
            "available": inference_service is not None
        },
        "system_info": {
            "python_version": os.sys.version,
            "evidently_available": EVIDENTLY_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )