import tensorflow as tf
from tensorflow.keras import layers, models
def create_face_recognition_model():
    model = models.Sequential()

    # Conv Block 1: 2 Conv layers with 64 filters
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # Pooling layer after first block

    # Conv Block 2: 2 Conv layers with 128 filters
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # Pooling layer after second block

    # Conv Block 3: 3 Conv layers with 256 filters
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # Pooling layer after third block

    # Conv Block 4: 3 Conv layers with 512 filters
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))  # Pooling layer after fourth block

    # Flatten layer to connect to Fully Connected (fc) layers
    model.add(layers.Flatten())

    # Fully Connected Layer 1: 4096 neurons
    model.add(layers.Dense(4096, activation='relu'))

    # Fully Connected Layer 2: 4096 neurons
    model.add(layers.Dense(4096, activation='relu'))

    # Output Layer: For face recognition, this would be the number of classes (individual identities)
    model.add(layers.Dense(1000, activation='softmax'))  # 1000 classes (adjust for the number of faces)

    return model

# Create the model
model = create_face_recognition_model()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.datasets import lfw

# Load LFW dataset (can be replaced with any other dataset)
# This is just an example, so if you have your own dataset, use it accordingly
(train_images, train_labels), (test_images, test_labels) = lfw.load_data()

# Preprocess images (resize and normalize)
train_images = tf.image.resize(train_images, (224, 224)) / 255.0
test_images = tf.image.resize(test_images, (224, 224)) / 255.0

# Make sure the images have the shape (224, 224, 3)
train_images = train_images[..., :3]
test_images = test_images[..., :3]
# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
# Make predictions on test data
predictions = model.predict(test_images)

# Example: Print the predicted class for the first test image
import numpy as np
print(f'Predicted class for first test image: {np.argmax(predictions[0])}')

