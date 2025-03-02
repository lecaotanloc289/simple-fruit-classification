import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Step 1: Define Dataset Path
dataset_path = "data/train"  
datatest_path = "data/test"  

# Step 2: Preprocess Data
img_size = (100, 100) 
batch_size = 32

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Load test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  
test_data = test_datagen.flow_from_directory(
    datatest_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  
)

# Get class names
class_names = list(train_data.class_indices.keys())
print("Classes:", class_names)

# Step 3: Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(class_names), activation="softmax") 
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Model Summary
model.summary()

# Step 4: Train Model
epochs = 10  

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# Step 5: Evaluate Model
val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.2f}, Validation Loss: {val_loss:.4f}")


# Step 6: Test Model 
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(test_data, steps=len(test_data)), axis=-1)
y_true = test_data.labels  

print(f"y_true length: {len(y_true)}, y_pred length: {len(y_pred)}") 
print(classification_report(y_true, y_pred, target_names=class_names))

# Show report classification
print(classification_report(y_true, y_pred, target_names=class_names))

# Step 7: Save Model
model.save("fruit_classifier2.h5")
print("Model saved as fruit_classifier.h5")

# Step 8: Plot Training Results
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()