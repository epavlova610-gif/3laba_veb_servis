# test_app.py
import io
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app  # ← твой файл называется main.py? (если lab3.py — поправь)

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
    assert "result_url" in response.text
    assert "original_url" in response.text


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
    assert "Обмен полос" in response.text or "result_url" in response.text


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