import tensorflow as tf
from tensorflow.keras import layers, models


def create_character_recognition_model():
    model = models.Sequential()

    # Convolutional Layer 1 (Conv1) - 5x5 kernel, 16 filters, stride=1
    model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'))

    # Max Pooling Layer 1 - 2x2 pool, stride=2
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # Convolutional Layer 2 (Conv2) - 5x5 kernel, 32 filters, stride=1
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))

    # Max Pooling Layer 2 - 2x2 pool, stride=2
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # Flatten Layer - to convert 2D matrices to 1D vector
    model.add(layers.Flatten())

    # Fully Connected Layer 1 (FC1) - 128 neurons
    model.add(layers.Dense(128, activation='relu'))

    # Fully Connected Layer 2 (FC2) - Output layer with 10 neurons (for 10 classes)
    model.add(layers.Dense(10, activation='softmax'))  # 10 classes for digits or characters

    return model
# Create the model
model = create_character_recognition_model()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Load MNIST dataset (digits 0-9)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Reshape images to 28x28x1 (single-channel grayscale) and normalize pixel values to [0, 1]
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
# Make predictions on test data
predictions = model.predict(test_images)

# Example: Print the predicted class for the first image in the test set
import numpy as np
print(f'Predicted class for first test image: {np.argmax(predictions[0])}')
