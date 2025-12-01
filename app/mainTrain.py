import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ✅ MAIN FOLDER PATH
dataset_path = r"C:\Users\phoen\OneDrive\Desktop\BrainTumourPredictionAPP\dataset"

NO_TUMOR_PATH = os.path.join(dataset_path, "no")
YES_TUMOR_PATH = os.path.join(dataset_path, "yes")

print("No folder exists:", os.path.exists(NO_TUMOR_PATH))
print("Yes folder exists:", os.path.exists(YES_TUMOR_PATH))

dataset = []
labels = []

# ✅ Function to load images
def load_images(folder_path, label):
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize
            image = cv2.resize(image, (64, 64))

            # Normalize (IMPORTANT)
            image = image / 255.0

            dataset.append(image)
            labels.append(label)


# ✅ Load both categories
load_images(NO_TUMOR_PATH, 0)  # No tumor
load_images(YES_TUMOR_PATH, 1) # Tumor

# Convert to numpy arrays
dataset = np.array(dataset)
labels = np.array(labels)

print("\nDataset shape:", dataset.shape)
print("Labels shape:", labels.shape)

# ✅ Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    dataset, labels, test_size=0.2, random_state=42, shuffle=True
)

# ✅ One-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test  = to_categorical(y_test, num_classes=2)

# ✅ BUILD CNN MODEL
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# ✅ Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ✅ Train model
history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=15,
    validation_data=(X_test, y_test),
    verbose=1
)

# ✅ Save model
model.save("BrainTumorFixed.h5")

print("\n✅ Model training complete and saved as BrainTumorFixed.h5")
