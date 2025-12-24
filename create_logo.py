from PIL import Image, ImageDraw, ImageFont

# Создаём изображение с прозрачным фоном
img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Рисуем круг (логотип)
draw.ellipse([20, 20, 180, 180], fill=(52, 152, 219, 255), outline=(41, 128, 185, 255), width=5)

# Добавляем текст в центр
try:
    font = ImageFont.truetype('arial.ttf', 80)
except:
    font = ImageFont.load_default()

text = 'WM'
bbox = draw.textbbox((0, 0), text, font=font)
text_x = (200 - (bbox[2] - bbox[0])) // 2
text_y = (200 - (bbox[3] - bbox[1])) // 2
draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

# Сохраняем
img.save('static/watermark_logo.png')
print('✅ Логотип создан: static/watermark_logo.png')
print('💡 Вы можете заменить этот файл своим логотипом (PNG с прозрачностью)')
