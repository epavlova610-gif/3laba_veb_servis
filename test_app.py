# test_app.py
import io
from PIL import Image
import numpy as np
from unittest.mock import patch

# Создаем клиент для тестирования
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def create_test_image(size=(100, 100), color=(255, 0, 0)):
    """Создаёт простое RGB-изображение в памяти."""
    img = Image.new("RGB", size, color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>" in response.text


def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200


def test_process_image_horizontal():
    img_bytes = create_test_image()
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {
        "direction": "horizontal",
        "strip_width": "10",
    }

    with patch("main.TF_AVAILABLE", False):  # отключаем TF в тестах
        response = client.post("/", data=data, files=files)

    assert response.status_code == 200
    assert "result_" in response.text
    assert "original_" in response.text


def test_process_image_vertical():
    img_bytes = create_test_image(color=(0, 255, 0))
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {
        "direction": "vertical",
        "strip_width": "20",
    }

    with patch("main.TF_AVAILABLE", False):
        response = client.post("/", data=data, files=files)

    assert response.status_code == 200
    assert "result_" in response.text


def test_swap_stripes_function():
    # Тест логики без FastAPI
    from main import swap_stripes

    # Создаём простой массив 4×4: [[0,1], [2,3]] по полосам ширины 2
    arr = np.array([
        [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
        [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0]],
        [[0, 0, 255], [0, 0, 255], [255, 255, 0], [255, 255, 0]],
        [[0, 0, 255], [0, 0, 255], [255, 255, 0], [255, 255, 0]],
    ], dtype=np.uint8)  # shape (4,4,3)

    result = swap_stripes(arr, direction="vertical", strip_width=2)

    # После обмена первые 2 столбца должны стать зелёными/жёлтыми, а вторые — красными/синими
    # Проверим, что (0,0) пиксель стал зелёным, а (0,2) — красным → обмен произошёл
    assert np.array_equal(result[0, 0], [0, 255, 0])  # раньше был [255,0,0]
    assert np.array_equal(result[0, 2], [255, 0, 0])  # раньше был [0,255,0]


def test_process_returns_response():
    """Тест проверяет, что POST-запрос обработки изображения возвращает успешный ответ"""
    img_bytes = create_test_image(size=(200, 200), color=(100, 150, 200))
    files = {"file": ("test_image.jpg", img_bytes, "image/jpeg")}
    data = {
        "direction": "horizontal",
        "strip_width": "15",
    }

    with patch("main.TF_AVAILABLE", False):
        response = client.post("/", data=data, files=files)

    # Проверяем, что запрос успешен
    assert response.status_code == 200
    # Проверяем, что в ответе есть ключевые элементы
    assert response.text is not None
    assert len(response.text) > 0
    # Проверяем наличие URL обработанного изображения
    assert "result_" in response.text or "static/" in response.text
    assert "result_" in response.text or "static/" in response.text


def test_watermark_added_to_processed_image():
    """Тест проверяет, что обработанное изображение содержит водяной знак"""
    from main import add_watermark
    
    # Создаём тестовое изображение
    test_img = Image.new("RGB", (300, 300), (255, 255, 255))
    
    # Добавляем водяной знак
    watermarked_img = add_watermark(test_img, "Обработано")
    
    # Проверяем, что изображение не None
    assert watermarked_img is not None
    # Проверяем, что размер не изменился
    assert watermarked_img.size == test_img.size
    # Проверяем, что изображение изменилось (водяной знак добавлен)
    assert watermarked_img.tobytes() != test_img.tobytes()


def test_endpoint_returns_valid_data():
    """Тест проверяет, что эндпоинт возвращает валидные данные с изображениями и гистограммой"""
    img_bytes = create_test_image(size=(150, 150), color=(50, 100, 150))
    files = {"file": ("test_valid.jpg", img_bytes, "image/jpeg")}
    data = {
        "direction": "vertical",
        "strip_width": "10",
    }

    with patch("main.TF_AVAILABLE", False):
        response = client.post("/", data=data, files=files)

    # Проверяем успешность запроса
    assert response.status_code == 200
    
    # Проверяем, что в ответе есть ссылки на все необходимые файлы
    assert "original_" in response.text  # Оригинальное изображение
    assert "result_" in response.text    # Обработанное изображение
    assert "histogram_" in response.text # Гистограмма