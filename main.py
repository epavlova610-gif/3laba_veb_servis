import hashlib
import io
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib
matplotlib.use('Agg')  # важно для работы без GUI
import matplotlib.pyplot as plt

# Инициализация приложения
app = FastAPI(title="Lab3: Вариант 17 — Обмен полос", description="Обмен чётных и нечётных полос + гистограмма RGB")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Создаём папку static при запуске
os.makedirs("static", exist_ok=True)


# --- Опциональная загрузка модели классификации (лениво) ---
TF_AVAILABLE = False
CLASSIFIER_MODEL = None

def try_load_tf():
    global TF_AVAILABLE, CLASSIFIER_MODEL
    if TF_AVAILABLE or CLASSIFIER_MODEL is not None:
        return
    try:
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
        model = MobileNetV2(weights='imagenet')
        CLASSIFIER_MODEL = (model, preprocess_input, decode_predictions)
        TF_AVAILABLE = True
    except Exception:
        TF_AVAILABLE = False


def swap_stripes(img_array: np.ndarray, direction: str, strip_width: int) -> np.ndarray:
    """
    Меняет местами чётные и нечётные полосы:
    - direction = 'horizontal': по строкам (по высоте)
    - direction = 'vertical': по столбцам (по ширине)
    """
    h, w = img_array.shape[:2]
    result = img_array.copy()

    if direction == "horizontal":
        num_strips = h // strip_width
        for i in range(0, num_strips - 1, 2):
            top1 = i * strip_width
            bot1 = (i + 1) * strip_width
            top2 = (i + 1) * strip_width
            bot2 = (i + 2) * strip_width
            if bot2 <= h:
                # Обмен с копированием, чтобы избежать aliasing
                result[top1:bot1], result[top2:bot2] = (
                    img_array[top2:bot2].copy(),
                    img_array[top1:bot1].copy(),
                )
    else:  # vertical
        num_strips = w // strip_width
        for i in range(0, num_strips - 1, 2):
            left1 = i * strip_width
            right1 = (i + 1) * strip_width
            left2 = (i + 1) * strip_width
            right2 = (i + 2) * strip_width
            if right2 <= w:
                result[:, left1:right1], result[:, left2:right2] = (
                    img_array[:, left2:right2].copy(),
                    img_array[:, left1:right1].copy(),
                )
    return result


def plot_histogram(img_array: np.ndarray, save_path: str):
    """
    Строит и сохраняет гистограмму распределения цветов (RGB) исходного изображения.
    """
    plt.figure(figsize=(6, 4))
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist, _ = np.histogram(img_array[:, :, i].flatten(), bins=256, range=(0, 256))
        plt.plot(hist, color=color, label=f'{color.upper()}')
    plt.title("Гистограмма распределения цветов (RGB)\n(исходное изображение)")
    plt.xlabel("Интенсивность (0–255)")
    plt.ylabel("Частота пикселей")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def process_image(
    request: Request,
    direction: str = Form(...),
    strip_width: int = Form(..., ge=1, le=500),
    file: UploadFile = File(...),
    classify: Optional[str] = Form(None),  # если checkbox отмечен — будет "on", иначе None
):
    try:
        # --- 1. Чтение и сохранение исходного изображения ---
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Файл пустой")

        # Генерируем уникальное имя по хешу содержимого
        hash_name = hashlib.md5(contents).hexdigest()[:10]
        original_path = f"static/original_{hash_name}.jpg"
        with open(original_path, "wb") as f:
            f.write(contents)

        # --- 2. Конвертация в RGB и массив numpy ---
        try:
            img = Image.open(original_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Некорректное изображение: {e}")
        img_array = np.array(img)

        # --- 3. Обмен полос ---
        processed_array = swap_stripes(img_array, direction, strip_width)
        processed_img = Image.fromarray(processed_array.astype("uint8"))
        result_path = f"static/result_{hash_name}.jpg"
        processed_img.save(result_path)

        # --- 4. Гистограмма ТОЛЬКО для исходного изображения ---
        hist_path = f"static/histogram_{hash_name}.png"
        plot_histogram(img_array, hist_path)

        # --- 5. Классификация (если запрошена) ---
        classification = None
        if classify:  # checkbox отмечен → classify == "on"
            try_load_tf()
            if TF_AVAILABLE and CLASSIFIER_MODEL is not None:
                model, preprocess_input, decode_predictions = CLASSIFIER_MODEL
                # Подготовка изображения для MobileNetV2
                pil_img = Image.open(original_path).convert('RGB')
                pil_resized = pil_img.resize((224, 224))
                arr = np.array(pil_resized).astype('float32')
                x = np.expand_dims(arr, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x, verbose=0)  # verbose=0 → без логов в консоль
                decoded = decode_predictions(preds, top=3)[0]
                classification = [(label, float(score)) for (_, label, score) in decoded]
            else:
                classification = [("TensorFlow не установлен", 0.0)]

        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_url": f"/static/original_{hash_name}.jpg",
            "result_url": f"/static/result_{hash_name}.jpg",
            "hist_url": f"/static/histogram_{hash_name}.png",
            "now": datetime.now(),
            "classification": classification,
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка обработки: {str(e)}")


# Для локального запуска
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)