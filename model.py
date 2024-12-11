import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(4096, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
