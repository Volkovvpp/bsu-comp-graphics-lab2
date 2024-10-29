import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

original_image = None
current_image = None
image_on_canvas = None

def select_image():
    global original_image, current_image, image_on_canvas, canvas

    img_path = filedialog.askopenfilename()

    if len(img_path) > 0:
        image = cv2.imread(img_path)
        if image is not None:
            original_image = image.copy()
            current_image = image.copy()

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)

            if image_on_canvas is not None:
                canvas.delete(image_on_canvas)

            image_on_canvas = canvas.create_image(0, 0, anchor='nw', image=image_tk)
            canvas.image = image_tk
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
        else:
            print("Ошибка загрузки изображения. Пожалуйста, выберите другое изображение.")


# Линейное контрастирование и поэлементные операции
def linear_contrast():
    global current_image, image_on_canvas, canvas
    if current_image is not None:
        b_channel, g_channel, r_channel = cv2.split(current_image)
        b_channel = cv2.add(b_channel, 50)
        g_channel = cv2.add(g_channel, 50)
        r_channel = cv2.add(r_channel, 50)

        b_channel = cv2.multiply(b_channel, 1.2)
        g_channel = cv2.multiply(g_channel, 1.2)
        r_channel = cv2.multiply(r_channel, 1.2)

        enhanced_image = cv2.merge((b_channel, g_channel, r_channel))
        f_image = enhanced_image.astype(np.float32)
        min_val = np.min(f_image)
        max_val = np.max(f_image)
        scaled_image = 255 * (f_image - min_val) / (max_val - min_val)
        scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)

        result_image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
        result_image_rgb = Image.fromarray(result_image_rgb)
        result_image_tk = ImageTk.PhotoImage(result_image_rgb)

        if image_on_canvas is not None:
            canvas.delete(image_on_canvas)
        image_on_canvas = canvas.create_image(0, 0, anchor='nw', image=result_image_tk)
        canvas.image = result_image_tk
    else:
        print("Нет изображения для обработки.")


# Метод Оцу для пороговой обработки
def otsu_threshold():
    global current_image, image_on_canvas
    if current_image is not None:
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result_image = Image.fromarray(otsu_thresh)
        result_image_tk = ImageTk.PhotoImage(result_image)

        if image_on_canvas is not None:
            canvas.delete(image_on_canvas)
        image_on_canvas = canvas.create_image(0, 0, anchor='nw', image=result_image_tk)
        canvas.image = result_image_tk
    else:   
        print("Нет изображения для обработки.")


def gradient_threshold():
    global current_image, image_on_canvas
    if current_image is not None:
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

        # Calculate gradients using Sobel operator
        grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

        # Calculate the gradient magnitude
        gradient_magnitude = np.maximum(np.abs(grad_x), np.abs(grad_y))

        # Compute the threshold t according to the formula
        numerator = np.sum(gradient_magnitude * gray_image)
        denominator = np.sum(gradient_magnitude)
        t = numerator / denominator if denominator != 0 else 0
        print(t)

        # Apply the threshold
        _, gradient_thresh = cv2.threshold(gradient_magnitude, t, 255, cv2.THRESH_BINARY)

        # Convert and display the result on the canvas
        result_image = Image.fromarray(np.uint8(gradient_thresh))
        result_image_tk = ImageTk.PhotoImage(result_image)

        if image_on_canvas is not None:
            canvas.delete(image_on_canvas)
        image_on_canvas = canvas.create_image(0, 0, anchor='nw', image=result_image_tk)
        canvas.image = result_image_tk
    else:
        print("Нет изображения для обработки.")


# Сброс изображения в исходное состояние
def reset_image():
    global original_image, current_image, image_on_canvas
    if original_image is not None:
        current_image = original_image.copy()
        display_image(current_image)


# Отображение изображения на холсте
def display_image(image):
    global image_on_canvas
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    if image_on_canvas is not None:
        canvas.delete(image_on_canvas)
    image_on_canvas = canvas.create_image(0, 0, anchor='nw', image=image_tk)
    canvas.image = image_tk


# Создание основного окна
root = tk.Tk()
root.title("Image processing")

title = tk.Label(root, text="Обработка изображений с помощью OpenCV", font=("Helvetica", 16))
title.pack(padx=10, pady=10)

btn_select = tk.Button(root, text="Выбрать изображение", command=select_image)
btn_select.pack(padx=10, pady=10)

canvas = tk.Canvas(root, width=500, height=500)
canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

v_scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

h_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

canvas.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

btn_contrast = tk.Button(root, text="Поэлементные операции\nи линейное контрастирование", command=linear_contrast)
btn_contrast.pack(side="left", padx=10, pady=10)

btn_otsu = tk.Button(root, text="Метод Оцу", command=otsu_threshold)
btn_otsu.pack(side="left", padx=10, pady=10)

btn_gradient = tk.Button(root, text="Градиентная пороговая обработка", command=gradient_threshold)
btn_gradient.pack(side="left", padx=10, pady=10)

btn_reset = tk.Button(root, text="Сбросить изображение", command=reset_image)
btn_reset.pack(side="bottom", padx=10, pady=10)

root.mainloop()