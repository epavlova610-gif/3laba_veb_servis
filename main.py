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
app.mount("/static", StaticFiles(directory="static"), name="static") #статические файлы серверов
templates = Jinja2Templates(directory="templates")

# Создаём папку static при запуске
os.makedirs("static", exist_ok=True)


# --- Опциональная загрузка модели классификации (лениво, модель загружается только при первом запросе с классификацией, чтобы не тормозить запуск) ---
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


def swap_stripes(img_array: np.ndarray, direction: str, strip_width: int) -> np.ndarray: # динамическая часть, разбивает изображение на полосы шириной strip_width по высоте или ширине
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


def plot_histogram(img_array: np.ndarray, save_path: str): # динаимическая часть, работает с загруженными файлами
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
    plt.close() # освобождает память


def add_watermark(img: Image.Image, text: str = "Обработано", logo_path: Optional[str] = None) -> Image.Image:
    """
    Добавляет водный знак на изображение.
    Можно использовать текст, изображение-логотип или оба варианта.
    
    Args:
        img: Исходное изображение
        text: Текст водяного знака
        logo_path: Путь к изображению-логотипу (опционально)
    """
    img_with_watermark = img.copy()
    width, height = img_with_watermark.size
    
    # Если указан путь к логотипу, пытаемся его загрузить
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            
            # Масштабируем логотип (максимум 15% от ширины изображения)
            logo_max_width = width // 7
            logo_aspect = logo.height / logo.width
            logo_width = min(logo.width, logo_max_width)
            logo_height = int(logo_width * logo_aspect)
            logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            
            # Создаём полупрозрачный вариант логотипа
            logo_with_alpha = logo.copy()
            alpha = logo_with_alpha.split()[3]  # Получаем альфа-канал
            alpha = alpha.point(lambda p: int(p * 0.7))  # Делаем на 70% прозрачнее
            logo_with_alpha.putalpha(alpha)
            
            # Позиция логотипа в правом нижнем углу
            logo_position = (width - logo_width - 15, height - logo_height - 15)
            
            # Накладываем логотип
            img_with_watermark.paste(logo_with_alpha, logo_position, logo_with_alpha)
            
            # Если есть текст, размещаем его над логотипом
            if text:
                draw = ImageDraw.Draw(img_with_watermark)
                try:
                    font = ImageFont.truetype("arial.ttf", size=max(16, width // 25))
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Текст над логотипом
                text_position = (width - text_width - 15, height - logo_height - text_height - 25)
                
                # Полупрозрачный фон для текста
                draw.rectangle(
                    [text_position[0] - 5, text_position[1] - 5,
                     text_position[0] + text_width + 5, text_position[1] + text_height + 5],
                    fill=(0, 0, 0, 128)
                )
                draw.text(text_position, text, fill=(255, 255, 255), font=font)
                
        except Exception as e:
            print(f"Ошибка загрузки логотипа: {e}")
            # Если не удалось загрузить логотип, добавляем только текст
            pass
    
    # Если логотип не указан или не загрузился, добавляем только текст
    if not logo_path or not os.path.exists(logo_path):
        draw = ImageDraw.Draw(img_with_watermark)
        
        try:
            font = ImageFont.truetype("arial.ttf", size=max(20, width // 20))
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        position = (width - text_width - 10, height - text_height - 10)
        
        draw.rectangle(
            [position[0] - 5, position[1] - 5,
             position[0] + text_width + 5, position[1] + text_height + 5],
            fill=(0, 0, 0, 128)
        )
        draw.text(position, text, fill=(255, 255, 255), font=font)
    
    return img_with_watermark


# роут - главная страница, Отдаёт форму загрузки (index.html)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


#роут пост - обработка изображения, файл читается целиком, обеспечена уникальность, 
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

        # --- 2. Конвертация в RGB и массив numpy,.convert("RGB") гарантирует 3 канала
        try:
            img = Image.open(original_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Некорректное изображение: {e}")
        img_array = np.array(img)

        # --- 3. Обмен полос,Результат сохраняется как JPEG.

        processed_array = swap_stripes(img_array, direction, strip_width)
        processed_img = Image.fromarray(processed_array.astype("uint8"))
        
        # Добавляем водный знак на обработанное изображение
        # Путь к логотипу можно изменить на свой
        logo_path = "static/watermark_logo.png"  # Создайте этот файл или укажите путь к своему логотипу
        processed_img = add_watermark(processed_img, text="Обработано", logo_path=logo_path)
        
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


        #возврат результата, должен отображать исходное изображение, обработанное, гистограмму, результат классификации
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