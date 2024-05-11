import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input data
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Set the limit for training and validation data
limit = 5000
x_train_data = x_train[:limit]
y_train_data = y_train_cat[:limit]
x_valid = x_train[limit:limit*2]
y_valid = y_train_cat[limit:limit*2]

# Add channel dimension
x_train_data = np.expand_dims(x_train_data, axis=3)
x_valid = np.expand_dims(x_valid, axis=3)

# Define the model architecture
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    #Specially 300 neurons for visual demonstration
    Dense(300, activation='relu'),
    #BatchNormalization, or dropout
    Dropout(0.8),
    #BatchNormalization(),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train_data, y_train_data, epochs=50, batch_size=32, validation_data=(x_valid, y_valid))

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()