import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from PIL import  Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os

image_width = 180
image_height = 180

import pathlib
model = tf.keras.models.load_model("C:\\Saved_models\\Model")




test_dir = "C:\\Varya\\validation"

images = os.listdir(test_dir)
X = []
y = []
for image_name in images:
    img_path = os.path.join(test_dir, image_name)
    img = Image.open(img_path)
    img = img.resize((image_width, image_height))
    img = np.array(img) / 255.0  # Нормализация изображения
    X.append(img)


X = np.array(X)
y = np.zeros(len(X),dtype=int)

# Получение предсказаний
predictions = model.predict(X)

# Рассчитать точность
accuracy = np.mean(np.argmax(predictions, axis=1) == y)

# Вывод точности
print("Точность на отложенном датасете:", accuracy)

# Визуализация точности
plt.bar(["Отложенный датасет"], [accuracy], color='green')
plt.ylabel('Точность')
plt.ylim([0, 1])  # Установить предел по оси Y от 0 до 1
plt.show()
