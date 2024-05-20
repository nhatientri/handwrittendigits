import tensorflow as tf


# Normalize the data
print("normalize data")
x_train = tf.keras.utils.normalize(x_train, axis=1)
print("normalize data complete")

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),  # Flatten the input
    tf.keras.layers.Dense(128, activation='relu'),  # First hidden layer
    tf.keras.layers.Dense(128, activation='relu'),  # Second hidden layer
    tf.keras.layers.Dense(62, activation='softmax')  # Output layer for 62 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128)

(x_test, y_test) = emnist.extract_test_samples('byclass')
print("Load test data")
x_test = tf.keras.utils.normalize(x_test, axis=1)
print("normalize data complete")
# Save the model
model.save('handwritten.keras')
