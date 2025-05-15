"""
Normalisation visual comparison of hand landmarks
This script captures video from the webcam, processes it using Mediapipe to detect hand landmarks,
and visualizes the original and normalized landmarks side by side.
It uses the `normalise_landmarks` function from the `data_gathering` module to normalize the landmarks.
"""

from data_gathering import normalise_landmarks
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)
    landmarks = np.zeros((42, 3))
    normalized_landmarks = np.zeros((42, 3))

    if results.multi_hand_landmarks:
        for index, hand in enumerate(results.multi_hand_landmarks):
            # Extract handedness and landmarks
            handedness = results.multi_handedness[index].classification[0].label
            hand_landmarks = hand.landmark
            if handedness == 'Right':
                landmarks[0:21] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            else:
                landmarks[21:42] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

        # Normalize landmarks
        normalized_landmarks = normalise_landmarks(landmarks)

    # Plot original landmarks with connections
    axs[0].cla()  # Clear previous plot
    axs[0].scatter(landmarks[:, 0], landmarks[:, 1], c='blue')
    axs[0].invert_yaxis()
    axs[0].plot(0,0)
    axs[0].set_title("Original Landmarks")
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        axs[0].plot(
            [landmarks[start_idx, 0], landmarks[end_idx, 0]],
            [landmarks[start_idx, 1], landmarks[end_idx, 1]],
            c='blue'
        )

    # Plot normalized landmarks with connections
    axs[1].cla()  # Clear previous plot
    axs[1].scatter(normalized_landmarks[:, 0], normalized_landmarks[:, 1], c='red')
    axs[1].invert_yaxis()
    axs[1].plot(0,0)
    axs[1].set_title("Normalized Landmarks")
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        axs[1].plot(
            [normalized_landmarks[start_idx, 0], normalized_landmarks[end_idx, 0]],
            [normalized_landmarks[start_idx, 1], normalized_landmarks[end_idx, 1]],
            c='red'
        )

    # Display the plots
    plt.pause(0.01)

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
plt.show()
cap.release()
cv2.destroyAllWindows()