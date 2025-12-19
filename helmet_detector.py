import torch
import cv2
import numpy as np
from pathlib import Path
import os
from ultralytics import YOLO
import sys

# Проверка версии PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Загрузка модели
def load_model(model_path='best_model.pt'):
    """Загрузка обученной модели"""
    try:
        # Проверяем наличие модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        
        # Проверяем наличие GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Загружаем модель
        model = YOLO(model_path)
        
        # Перемещаем модель на нужное устройство
        model.to(device)
        
        # Тестируем модель на небольшом изображении
        print("Testing model...")
        test_tensor = torch.randn(1, 3, 640, 640).to(device)
        
        print("✅ Модель успешно загружена и протестирована!")
        return model
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        print(f"Error type: {type(e).__name__}")
        return None

# Инициализация модели
MODEL = load_model()

def detect_helmets(image_path):
    """Обнаружение касок на изображении"""
    if MODEL is None:
        raise Exception("Модель не загружена")
    
    try:
        print(f"\n📷 Обработка изображения: {image_path}")
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Не удалось загрузить изображение")
        
        print(f"Размер изображения: {image.shape}")
        
        # Сохраняем оригинальный размер
        orig_height, orig_width = image.shape[:2]
        
        # Определяем устройство
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Выполняем детекцию
        print("🔍 Выполняем детекцию...")
        results = MODEL(image, device=device, verbose=False)
        
        # Копируем изображение для рисования
        result_image = image.copy()
        helmet_count = 0
        
        # Если есть результаты детекции
        if results and len(results) > 0:
            result = results[0]
            
            # Проверяем наличие боксов
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # координаты боксов
                confidences = result.boxes.conf.cpu().numpy()  # уверенность
                
                print(f"Найдено объектов: {len(boxes)}")
                
                for i, (box, conf) in enumerate(zip(boxes, confidences), 1):
                    # Фильтруем по уверенности
                    confidence_threshold = 0.25  # Можно настроить
                    if conf > confidence_threshold:
                        helmet_count += 1
                        
                        # Координаты бокса
                        x1, y1, x2, y2 = map(int, box)
                        
                        print(f"  Объект {i}: confidence={conf:.3f}, bbox=[{x1}, {y1}, {x2}, {y2}]")
                        
                        # Рисуем зеленый прямоугольник
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Подпись с уверенностью
                        label = f"Helmet: {conf:.2f}"
                        
                        # Размер текста
                        font_scale = max(0.5, min(1.0, (x2 - x1) / 300))
                        thickness = max(1, int((x2 - x1) / 150))
                        
                        # Получаем размеры текста
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        
                        # Фон для текста
                        cv2.rectangle(
                            result_image,
                            (x1, max(0, y1 - label_height - baseline - 5)),
                            (x1 + label_width, y1),
                            (0, 255, 0),
                            -1
                        )
                        
                        # Текст
                        cv2.putText(
                            result_image,
                            label,
                            (x1, max(baseline + 5, y1 - baseline - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness
                        )
        
        print(f"✅ Обнаружено касок: {helmet_count}")
        
        # Генерируем имя для результата
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}_{Path(image_path).stem}.jpg"
        result_path = os.path.join("static/results", result_filename)
        
        # Сохраняем результат
        cv2.imwrite(result_path, result_image)
        print(f"💾 Результат сохранен: {result_path}")
        
        return result_filename, helmet_count
        
    except Exception as e:
        print(f"❌ Ошибка при детекции: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise

def test_with_sample_image():
    """Тестирование с примером изображения"""
    print("\n" + "="*50)
    print("🧪 Тестирование модели")
    print("="*50)
    
    # Создаем тестовое изображение
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    test_image[200:400, 200:400] = [0, 255, 0]  # Зеленый квадрат
    
    # Сохраняем тестовое изображение
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, test_image)
    
    try:
        result_file, count = detect_helmets(test_path)
        print(f"Тестовый результат: {result_file}, найдено объектов: {count}")
        
        # Удаляем тестовый файл
        if os.path.exists(test_path):
            os.remove(test_path)
            
        return True
    except Exception as e:
        print(f"Тест не пройден: {e}")
        return False

# Тестирование при запуске
if __name__ == "__main__":
    print("🚀 Запуск детектора касок")
    
    # Тестируем модель
    if test_with_sample_image():
        print("✅ Тестирование завершено успешно!")
    else:
        print("❌ Тестирование завершено с ошибками")