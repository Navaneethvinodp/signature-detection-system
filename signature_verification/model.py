import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
DATASET_PATH = 'D:\\Aaa Class\\sem_7\\biometrics\\archive (9)\\CEDAR\\CEDAR'
IMG_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32
EPOCHS = 10

# Function to load and preprocess images
def load_images(dataset_path):
    images = []
    labels = []
    
    for person_id in range(1, 56):  # Loop through folders 1 to 55
        person_folder = os.path.join(dataset_path, str(person_id))
        
        if not os.path.exists(person_folder):
            print(f"Folder {person_folder} does not exist!")
            continue
        
        for filename in os.listdir(person_folder):
            if filename.startswith('forgeries'):  # Forged signatures
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)  # Resize to standard size
                images.append(img)
                labels.append(0)  # Label for forged signatures
            elif filename.startswith('original'):  # Genuine signatures
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(1)  # Label for genuine signatures
    
    return np.array(images), np.array(labels)

# Load images and preprocess them
images, labels = load_images(DATASET_PATH)
images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0  # Normalize images
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create a data generator for training
train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                    shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
                                    fill_mode='nearest')

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
          validation_data=(X_val, y_val),
          epochs=EPOCHS)

# Save the model
model.save('signature_model.h5')
