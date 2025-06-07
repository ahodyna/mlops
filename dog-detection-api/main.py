from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
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

inference_service = None

@app.on_event("startup")
async def startup_event():
    """Ініціалізація сервісу при запуску"""
    global inference_service
    try:
        mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        minio_bucket = os.getenv("MINIO_BUCKET", "mlflow-artifacts")
        
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
        raise

@app.get("/")
async def root():
    """Головна сторінка API"""
    return {
        "message": "🐕 Dog Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "batch_predict": "/batch-predict"
        }
    }

@app.get("/health")
async def health_check():
    """Перевірка стану сервісу"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "dog-detection-api"
    }

@app.get("/models")
async def list_models():
    """Отримати список доступних моделей"""
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
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл повинен бути зображенням")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        inference_service.confidence_threshold = confidence
        inference_service.iou_threshold = iou
        
        model = inference_service.load_model(model_path)
        if model is None:
            raise HTTPException(status_code=500, detail="Не вдалося завантажити модель")
        
        result = inference_service.run_inference_on_image(
            model, 
            Path(temp_path), 
            save_results=False
        )
        
        os.unlink(temp_path)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Помилка обробки зображення")
        
        return {
            "filename": file.filename,
            "detections": result['detections'],
            "detection_count": result['detection_count'],
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Помилка інференсу: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    model_path: Optional[str] = Query(None)
):
    """Batch інференс на декількох зображеннях"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Максимум 10 файлів за раз")
    
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"Файл {file.filename} не є зображенням")
    
    try:
        inference_service.confidence_threshold = confidence
        inference_service.iou_threshold = iou
        
        model = inference_service.load_model(model_path)
        if model is None:
            raise HTTPException(status_code=500, detail="Не вдалося завантажити модель")
        
        results = []
        temp_files = []
        
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
        
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        total_detections = sum(r['detection_count'] for r in results)
        images_with_dogs = len([r for r in results if r['detection_count'] > 0])
        
        return {
            "results": results,
            "summary": {
                "total_images": len(files),
                "processed_images": len(results),
                "total_detections": total_detections,
                "images_with_dogs": images_with_dogs,
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        if 'temp_files' in locals():
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Помилка batch інференсу: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Статистика сервісу"""
    return {
        "service": "dog-detection-api",
        "model_info": {
            "latest_model": inference_service.get_latest_model() if inference_service else None,
            "confidence_threshold": inference_service.confidence_threshold if inference_service else None,
            "iou_threshold": inference_service.iou_threshold if inference_service else None
        },
        "system_info": {
            "python_version": os.sys.version,
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