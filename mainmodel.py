import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

img_width = 180
img_height = 180

def main():
    print('Выберите действие:\n'
          '1. Загрузить данные\n'
          '2. Построить модель\n'
          '3. Обучить\n'
          '4. аугментация данных\n'
          '5. Попробовать с изображением\n')
    cmd = str(input("Введите номер действия: "))
    while cmd:
        if cmd.lower() == '1':
            train_ds, val_ds, class_names, AUTOTUNE = data1()
        elif cmd.lower() == '2':
            model,num_classes = build(class_names, img_height, img_width)
        elif cmd.lower() == '3':
            educate(train_ds, val_ds, model, class_names, img_height, img_width)
        elif cmd.lower() == '4':
            augment(num_classes,val_ds,train_ds)
        elif cmd.lower() == '5':
            try_img(model, class_names, img_height, img_width)
        else:
            print('Некорректное действие. Попробуйте снова.')
        cmd = str(input("Введите номер действия: "))

def data1():
    dataset_dir = "C:\\Varya"
    batch_size = 32
    img_width = 180
    img_height = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(f"Class names: {class_names}")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names, AUTOTUNE,


def build(class_names, img_height, img_width):
    num_classes = len(class_names)
    model = Sequential([
        # т.к. у нас версия TF 2.6 локально
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

        # дальше везде одинаково
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # print model summary
    model.summary()
    return model, num_classes

def educate(train_ds, val_ds, model, class_names, img_height, img_width):
    epochs = 10  # количество эпох тренировки
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    # visualize training and validation results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def augment(num_classes,val_ds,train_ds):
    model = Sequential([
        # т.к. у нас версия TF 2.6 локально
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

        # аугментация
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.RandomContrast(0.2),

        # дальше везде одинаково
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        # регуляризация
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # print model summary
    model.summary()

    epochs = 20  # количество эпох тренировки
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    # visualize training and validation results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    model.save("C:\\Saved_models\\Model")
def try_img(model, class_names, img_height, img_width):
    #sunflower_url  = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    #sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    pathtest = "C:\\Try\\var.jpg"
    img = tf.keras.utils.load_img(
        pathtest, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # print inference result
    print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
        class_names[np.argmax(score)],
        100 * np.max(score)))

    # show the image itself
    img.show()

if __name__ == "__main__":
    main()

