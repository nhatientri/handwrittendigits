import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def savedata():
    # Import mnist directly from tensorflow
    mnist = tf.keras.datasets.mnist
    # Divide the training data and testing data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize the data into a value in the range of 0 - 1 instead of RGB 0 - 255
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Start creating the model
    model = tf.keras.models.Sequential([])
    model.add(tf.keras.layers.InputLayer(shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)
    model.save('handwritten.keras')


def loaddata():
    # Load the pre-trained model
    model = tf.keras.models.load_model('handwritten.keras')
    image_number = 0
    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            # Read the image in grayscale mode
            image = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)

            # Invert the image colors
            image = np.invert(np.array(image))

            # Resize the image to 28x28 pixels if it's not already that size
            if image.shape != (28, 28):
                image = cv2.resize(image, (28, 28))

            # Normalize the image
            image = image / 255.0

            # Reshape the image to match the input shape expected by the model (28, 28, 1)
            image = image.reshape(1, 28, 28, 1)

            # Predict the digit
            prediction = model.predict(image)
            predicted_digit = np.argmax(prediction)

            # Display the image
            plt.imshow(image.reshape(28, 28), cmap='gray')
            plt.title(f"Predicted: {predicted_digit}")
            plt.axis('off')  # Hide the axes
            plt.show()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            image_number += 1


if __name__ == '__main__':
    # savedata()
    loaddata()
