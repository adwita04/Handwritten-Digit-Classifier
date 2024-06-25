import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as utils

# Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = utils.normalize(x_train, axis=1)
x_test = utils.normalize(x_test, axis=1)

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=75)

# Predict digits from images in the "digits" directory
image_number = 1
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        # Read and preprocess the image
        img = cv2.imread(f"digits/{image_number}.png", cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image {image_number} could not be read.")

        img = np.invert(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=-1)  # Add channel dimension

        # Predict the digit
        prediction = model.predict(img)
        print(f"The number is {np.argmax(prediction)}")
        
        # Display the image
        plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
        plt.title(f"Predicted Number: {np.argmax(prediction)}")
        plt.show()

    except Exception as e:
        print(f"Error processing image {image_number}: {e}")

    finally:
        image_number += 1
