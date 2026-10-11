import os
import tensorflow as tf
from tensorflow.keras import layers

# Target directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "Xraymodel.h5")

print("Creating X-Ray CNN model architecture...")
model = tf.keras.Sequential([
    layers.Input(shape=(200, 200, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')  # 3 classes: covid19, normal, pneumonia
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Saving compiled model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("Xraymodel.h5 generated successfully!")
