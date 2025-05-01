
"""
Train a neural network model to predict hand shapes from landmarks.
Author: Ryan Bright
Date: --
Licence: --
Description:
Loads data from json files and trains a neural network model to predict hand shapes from landmarks.
This code is a part of a larger project that uses MediaPipe to gather hand landmarks and train a model 
to predict hand shapes. The model is then saved for later use.
Data augmentation is used to increase the size of the dataset and improve the model's performance.
"""
import numpy as np
import json
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore  (to remove import error that actually works)
from tensorflow.keras.callbacks import EarlyStopping
import pickle

def augment_landmarks(landmarks, shift_range=0.02, noise_std=0.005):
    """
    Apply random translation, and noise to each hand independently from a (126) landmark list.
    This is to add variation in joint locations *and* in relitive hand positions.
    """
    landmarks = np.array(landmarks).reshape(-1, 3)  # Reshape to (num_landmarks, 3)
    landmarks_to_vary = landmarks[landmarks != 0]

    # Random small Gaussian noise to each nonzero landmark
    noise = np.random.normal(0, noise_std, landmarks_to_vary.shape)
    landmarks_to_vary += noise
    
    

    # Random translations to each hand
    left_hand = landmarks[:63]  # Left hand nonzero landmarks
    right_hand = landmarks[63:]  # Right hand nonzero landmarks
    for hand in [left_hand, right_hand]:
        # Random translation in x, y, z directions
        shift_x = np.random.uniform(-shift_range, shift_range)
        shift_y = np.random.uniform(-shift_range, shift_range)
        shift_z = np.random.uniform(-shift_range, shift_range)
        
        hand[hand[:,0]!=0, 0] += shift_x
        hand[hand[:,1]!=0, 1] += shift_y
        hand[hand[:,2]!=0, 2] += shift_z
    

    return landmarks.reshape(-1).tolist()  # Flatten back to (126) list

X = []
y = []

datapath = 'landmarks_data/'  # Path to the folder containing JSON files

for filepath in glob.glob(f'{datapath}/*.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    landmarks = np.append(np.array(data['left_hand']).reshape(21*3), 
                          np.array(data['right_hand']).reshape(21*3)).tolist()
    # print(np.array(data['left_hand']).shape)  # (21, 3)
    # print(len(landmarks))  # (126) = (21*3 + 21*3)
    X.append(landmarks)
    y.append(data['label'])

# Data augmentation
for i in range(len(X)):
    # For each original landmark, create 5 augmented landmarks
    if y[i] != 'Z':  # Dont augment 'Z' labels (there is already lots)
        for j in range(200):
            # Augment the data by applying random transformations
            augmented_landmarks = augment_landmarks(X[i])
            X.append(augmented_landmarks)
            y.append(y[i])  # Append the same label for the augmented data

X = np.array(X)
y = np.array(y)
# print(X.shape, y.shape)  # (num_samples, 126), (num_samples,)

# Encode text labels into integer indexes
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model definitions 
num_classes = len(le.classes_)  # Number of classes (NZSL signs)

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def MLP():
    """
    Define a simple MLP model.
    """
    model = tf.keras.models.Sequential([
        layers.Input(shape=(126,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = MLP()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate model
print('Evaluating model...')
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f} over {len(X_test)} samples')

# Save model
model.save('models_and_encoders/hand_sign_model.keras')
# Save label encoder too (important for decoding predictions later!)
with open('models_and_encoders/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
