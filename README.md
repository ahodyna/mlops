# 🐶 Виявлення собаки на зображеннях (Object Detection)

Цей репозиторій містить інструменти для розмітки зображень із собаками, зберігання даних і підготовки їх до навчання моделі виявлення об'єктів.

---

## 📌 Компоненти системи

- **🏷️ Розмітка даних**: [Label Studio](https://labelstud.io/) - веб-інтерфейс для анотації зображень
- **🧠 Модель**: YOLOv8 - сучасна архітектура для детекції об'єктів
- **📊 Трекінг експериментів**: MLflow - версіонування та моніторинг моделей
- **💾 Сховище**: MinIO - S3-сумісне сховище для даних та артефактів
- **🔧 Версіонування даних**: DVC - контроль версій датасетів
- **🚀 API**: FastAPI - REST API для інференсу моделі
- **📈 Моніторинг**: Prometheus + Grafana - метрики та візуалізація
- **🐳 Оркестрація**: Docker Compose - контейнеризація всіх сервісів

## 1. Запустіть docker-compose файл:

     docker-compose up


## 2. Створіть облікові записи

- **Label Studio**: [http://localhost:8080](http://localhost:8080)  
- **MLflow**: [http://localhost:5001](http://localhost:5001)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)  
- **Grafana**: [http://localhost:3000](http://localhost:3000)
- **Dog Detection API**: [http://localhost:8001](http://localhost:8001)
- **MinIO**: [http://localhost:9009](http://localhost:9009)  
  > 🔐 Дані для входу вказані у `docker-compose.yml`.

---

## 3. Налаштуйте MinIO

1. Створіть бакет.
2. Завантажте зображення для розмітки.

---

## 4. Налаштуйте Label Studio

1. Створіть новий проєкт.
2. Перейдіть у `Settings → Cloud Storage`.
3. Додайте підключення до вашого бакета MinIO.
4. Натисніть **Sync Storage**, щоб імпортувати зображення.

---

## 5. Розмітка даних

1. Відмічайте зони із собаками на зображеннях.
2. Після завершення розмітки експортуйте результати (наприклад, у форматі **YOLO**).

---

## 📂 Версіонування даних із DVC

### 1. Підготуйте структуру

- Створіть папку `dataset`.
- Додайте розмічені дані, які експортували з Label Studio.

### 2. Налаштуйте DVC

1. Активуйте Python-віртуальне середовище.
2. Встановіть DVC:

   ```bash
   pip install dvc
   dvc add dataset
   
   git add .
   commit -m "New version of labeled data"
   dvc push

    ```

   ## 🤖 Тренування моделі

### 3. Параметри тренування

| Параметр | Опис | Значення за замовчуванням |
|----------|------|---------------------------|
| `--model` | Розмір YOLOv8 моделі | `yolov8s.pt` |
| `--epochs` | Кількість епох | `50` |
| `--batch` | Розмір batch | `2` |
| `--lr0` | Learning rate | `0.0001` |
| `--imgsz` | Розмір зображень | `416` |

### Підготовка даних

```bash
# Перемаплення датасету 
python create_number_mapping.py
```

### Запуск тренування

```bash
python src/train_yolo.py --epochs 5 --batch 4
```

### Моніторинг
## 📊 Моніторинг та результати

### MLflow UI

Відкрийте [http://localhost:5001](http://localhost:5001) для перегляду:

- **Метрики**: mAP50, mAP50-95, Precision, Recall
- **Параметри**: всі гіперпараметри тренування
- **Артефакти**: графіки навчання, результати валідації
- **Моделі**: збережені checkpoint'и



---

## 🔮 Інференс та використання

### 1. Тестування через скрипт

```bash
# Інференс на всіх зображеннях з папки
python src/yolo_inference.py --test_dir test_images

# Тестування одного зображення
python src/yolo_inference.py --single_image path/to/image.jpg

# Використання конкретної моделі
python src/yolo_inference.py \
    --model models/dog_detection_20241215_12345/best.pt \
    --confidence 0.7 \
    --iou 0.5

# Список доступних моделей
python src/yolo_inference.py --list_models
```

### 2. REST API

#### Запуск API

```bash
# API автоматично запускається з docker-compose
# Або окремо:
cd dog-detection-api
python main.py
```

#### Приклади використання

```bash
# Перевірка статусу
curl http://localhost:8001/health

# Список доступних моделей
curl http://localhost:8001/models

# Детекція на одному зображенні
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  -F "confidence=0.6"

# Batch детекція
curl -X POST "http://localhost:8001/batch-predict" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

#### API Endpoints

| Endpoint | Метод | Опис |
|----------|-------|------|
| `/` | GET | Інформація про API |
| `/health` | GET | Статус сервісу |
| `/models` | GET | Список моделей |
| `/predict` | POST | Детекція на одному зображенні |
| `/batch-predict` | POST | Batch детекція (до 10 файлів) |
| `/stats` | GET | Статистика сервісу |

### 3. Результати інференсу

```json
{
  "filename": "dog_photo.jpg",
  "detections": [
    {
      "bbox": [245.3, 167.8, 456.2, 378.9],
      "confidence": 0.87,
      "class_id": 0,
      "class_name": "dog"
    }
  ],
  "detection_count": 1,
  "timestamp": "2024-12-15T14:30:25"
}
```

---

### Prometheus + Grafana
- **Метрики API**: `dog_predictions_total`, `dog_processing_seconds`, `dogs_detected_total`
- **Дашборд**: "Dog Detection API Dashboard" у Grafana

### YOLO ML Pipeline

Основний пайплайн включає три етапи:

1. **setup-infrastructure** - запуск MinIO, MLflow, DB з перевіркою готовності сервісів
2. **train** - встановлення залежностей, тренування YOLOv8 моделі (10 епох), збереження в MinIO
3. **deploy** - деплой API для інференсу з готовністю для тестування

```bash
# Запуск повного ML пайплайну
act 

```

## 🔮 Подальше використання
- Дані будуть використані для навчання та тестування моделі, яка дозволить автоматично виявляти собаку на нових фото.

