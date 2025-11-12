import hashlib
import os
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # без GUI
import matplotlib.pyplot as plt

app = FastAPI(title="Lab3: Variant 17")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("static", exist_ok=True)


def swap_stripes(img_array: np.ndarray, direction: str, strip_width: int) -> np.ndarray:
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
                # обмен полос
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
    plt.figure(figsize=(6, 4))
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist, _ = np.histogram(img_array[:, :, i].flatten(), bins=256, range=(0, 256))
        plt.plot(hist, color=color, label=f'{color.upper()}')
    plt.title("Гистограмма RGB (исходное изображение)")
    plt.xlabel("Интенсивность")
    plt.ylabel("Частота")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def process_image(
    request: Request,
    direction: str = Form(...),
    strip_width: int = Form(...),
    file: UploadFile = File(...),
):
    try:
        # 1. Сохраняем исходное изображение
        contents = await file.read()
        hash_name = hashlib.md5(contents).hexdigest()[:10]
        original_path = f"static/original_{hash_name}.jpg"
        with open(original_path, "wb") as f:
            f.write(contents)

        # 2. Читаем и конвертируем в RGB
        img = Image.open(original_path).convert("RGB")
        img_array = np.array(img)

        # 3. Обрабатываем: обмен полос
        processed_array = swap_stripes(img_array, direction, strip_width)
        processed_img = Image.fromarray(processed_array.astype("uint8"))
        result_path = f"static/result_{hash_name}.jpg"
        processed_img.save(result_path)

        # 4. Гистограмма — **только для исходного**
        hist_path = f"static/histogram_{hash_name}.png"
        plot_histogram(img_array, hist_path)

        # 5. Возвращаем результат
        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_url": f"/static/original_{hash_name}.jpg",
            "result_url": f"/static/result_{hash_name}.jpg",
            "hist_url": f"/static/histogram_{hash_name}.png",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")