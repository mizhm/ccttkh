import tensorflow as tf
from keras import layers, models


def create_model():
    model = models.Sequential()

    # Input Layer and 1st Convolutional Layer (Conv1)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))

    # 2nd Convolutional Layer (Conv2)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

    # Pooling Layer
    model.add(layers.MaxPooling2D((2, 2), padding='valid'))

    # Flatten Layer
    model.add(layers.Flatten())

    # Fully Connected Layer 1 (FC1)
    model.add(layers.Dense(128, activation='relu'))

    # Fully Connected Layer 2 (FC2) - Output Layer
    model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for digits 0-9

    return model


model = create_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
predictions = model.predict(test_images)

import numpy as np
print(f'Predicted class for first test image: {np.argmax(predictions[0])}')
