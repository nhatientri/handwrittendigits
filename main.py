import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import mnist directly from tensorflow
mnist = tf.keras.datasets.mnist
# Divide the training data and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the data into a value in the range of 0 - 1 instead of RGB 0 - 255
x_train, x_test = x_train / 255.0, x_test / 255.0

# Start creating the model
model = tf.keras.models.Sequential([])
# Using tf.keras.layers.InputLayer(input_shape=(28, 28, 1))
# specifies the input shape including the channel dimension (1 for grayscale images).
model.add(tf.keras.layers.InputLayer(shape=(28, 28, 1)))
# Flatten the layer (i.e a 28x28 grid turns into a 784x1 line
model.add(tf.keras.layers.Flatten())
# Add the hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Add the last layer to recognize the last layer
# Softmax is kinda similar to the sigmoid function
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.save('handwritten.keras')
