import cv2
import numpy as np
import mediapipe as mp

from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import LabelEncoder
from data_gathering import normalise_landmarks # Get normalise_landmarks function from data_gathering.py
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load model and labels
model = load_model('models_and_encoders/hand_sign_model.keras')
with open('models_and_encoders/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Matplotlib setup for live bar chart
plt.ion()
fig, ax = plt.subplots()
class_names = encoder.classes_
bar_container = ax.bar(class_names, [0] * len(class_names))
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')
ax.set_title('Class Probabilities')

def update_bar_chart(probabilities):
    for bar, prob in zip(bar_container, probabilities):
        bar.set_height(prob)
    fig.canvas.draw()
    fig.canvas.flush_events()

def preprocess_landmarks(landmarks):
    """
    Input: list of (42, 3) landmarks for two hands.
    Output: flattened and normalized (126,) vector.
    """
    return normalise_landmarks(landmarks).reshape(-1).tolist()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        landmarks = np.zeros((42, 3))
        for index, hand in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            # Extract handedness and landmarks
            handedness = results.multi_handedness[index].classification[0].label
            hand_landmarks = hand.landmark
            if handedness == 'Right':
                landmarks[0:21] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            else:
                landmarks[21:42] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
        input_data = preprocess_landmarks(landmarks)
        prediction = model.predict(np.array([input_data]), verbose=0)
        probabilities = prediction[0]
        class_id = np.argmax(probabilities)
        label = encoder.inverse_transform([class_id])[0]
        confidence = probabilities[class_id] * 100
        cv2.putText(frame, f'Predicted: {label} ({confidence:.2f}%)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        update_bar_chart(probabilities)
    cv2.imshow('Hand Sign Inference', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
