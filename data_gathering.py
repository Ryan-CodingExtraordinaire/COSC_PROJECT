"""
Data capture and hand detection functions. 

Author: Ryan Bright

Date: --

Licence: --

Description:
Using MediaPipe, the hand landmarks are extracted from the video stream.
The landmarks are normalised relative to the wrists and saved to a JSON file.
The JSON file is saved in the 'landmarks_data' directory with a timestamp and label.
The landmarks are saved in the following format:
    
    {
        'label': '<user key press>',
        'timestamp': 1234567890,
        'left_hand': [[x1, y1, z1], [x2, y2, z2], ...],
        'right_hand': [[x1, y1, z1], [x2, y2, z2], ...]
    }
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time

def normalise_landmarks(landmarks):
    """Normalise the landmarks relitive to the wrist and scaled to distance between wrists.
    Missing hands have all landmarks set to 0.
    If only one hand is present then the landmark values of each hand get added to each other
    makes both hands the same pose and the normalisation works to give translation symmetry
    To get scale symmetery, when there is only one hand the scale metric becomes the 
    wrist-to-fingertip distance as opposed to the wrist-to-wrist distance.

    Input: np.array of (42, 3) landmarks for two hands.
    Output: normalised (42, 3) landmarks for two hands."""

    # Establish which hands are present
    left_hand = landmarks[0:21].copy()
    right_hand = landmarks[21:42].copy()
    has_left = True
    has_right = True
    if np.all(left_hand != 0):
        has_left = False
    if np.all(right_hand != 0):
        has_right = False
    
    # If only one hand is present then make each hand have the same values
    if has_left != has_right: 
        landmarks[:21] += right_hand
        landmarks[21:] += left_hand
        
    # Choose origin: midpoint between wrists (landmarks[0] and landmarks[21])
    left_wrist = landmarks[0]
    right_wrist = landmarks[21]
    origin = (left_wrist + right_wrist) / 2
    # Center landmarks around the origin
    landmarks -= origin

    # Choose scale: farthest landmark distance from origin
    max_distance = np.max(np.linalg.norm(landmarks - origin, axis=1))
    landmarks /= np.abs(max_distance)
    return landmarks

def main():
    def show_landmark_data(filename):
        """Display the landmark data from a JSON file.
        Data is stored in the following format using lists:
        
        {
            'label': '<user key press>',
            'timestamp': 1234567890,
            'left_hand': [[x1, y1, z1], [x2, y2, z2], ...],
            'right_hand': [[x1, y1, z1], [x2, y2, z2], ...]
        }
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            
            # Extract left and right hand landmarks
            left_hand = np.array(data['left_hand'])
            right_hand = np.array(data['right_hand'])

            # # Invert the y-axis for correct display
            # left_hand[:][1] = -left_hand[:][1]
            # right_hand[:][1] = -right_hand[:][1]

            # Plot the landmarks
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            if left_hand.size > 0:
                # Plot left hand landmarks
                ax.scatter(left_hand[:, 0], left_hand[:, 1], left_hand[:, 2], c='blue', label='Left Hand')
                # Plot left hand connections
                left_hand_connections = mp_hands.HAND_CONNECTIONS
                for connection in left_hand_connections:
                    start_idx, end_idx = connection
                    ax.plot(
                        [left_hand[start_idx, 0], left_hand[end_idx, 0]],
                        [left_hand[start_idx, 1], left_hand[end_idx, 1]],
                        [left_hand[start_idx, 2], left_hand[end_idx, 2]],
                        c='blue'
                    )

            if right_hand.size > 0:
                # Plot right hand landmarks
                ax.scatter(right_hand[:, 0], right_hand[:, 1], right_hand[:, 2], c='red', label='Right Hand')
                # Plot right hand connections
                right_hand_connections = mp_hands.HAND_CONNECTIONS
                for connection in right_hand_connections:
                    start_idx, end_idx = connection
                    ax.plot(
                        [right_hand[start_idx, 0], right_hand[end_idx, 0]],
                        [right_hand[start_idx, 1], right_hand[end_idx, 1]],
                        [right_hand[start_idx, 2], right_hand[end_idx, 2]],
                        c='red'
                    )    
                    
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.show()

    # Create a directory for saving landmarks if it doesn't exist
    output_dir = "landmarks_data"
    os.makedirs(output_dir, exist_ok=True)

    # Setup MediaPipe model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Frame capture settings
    mode = "Display"
    label = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract hand landmarks from frame
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                # Create an array to hold the landmarks for both hands
                landmarks = np.zeros((42,3))

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
                # print(landmarks)

                if mode == "Train" and label:
                    landmarks_normalized = normalise_landmarks(landmarks)
                    timestamp = int(time.time() * 1000)  # Milliseconds
                    sample = {
                        'label': label,
                        'timestamp': timestamp,
                        'left_hand': landmarks_normalized[0:21].tolist(),
                        'right_hand': landmarks_normalized[21:42].tolist()
                    }

                    filename = f"{label}_{timestamp}.json"
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, 'w') as f:
                        json.dump(sample, f)
                    print(f"[Saved] Gesture '{label}' as {filename}")
                    # Reset label if it has been 10 seconds
                    if int(time.time()) - (starttime) > 5:
                        print(f"[Reset] Gesture '{label}'")
                        label = None
                    

        # Show the mode and label on the frame
        text = f"Mode: {mode}"
        if label:
            text += f" | Label: {label}"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # show the hand landmarks
        cv2.imshow('Hand Tracking', frame)
        
        key = cv2.waitKey(1)
        if mode == "Display":
            if key == 27:   # Wait for 'Esc' key to exit displaying loop
                break
            if key == ord('`'):
                mode = "Train"
        elif mode == "Train":
            if key == 27 or key == ord('`'):
                mode = "Display"
            if key >= ord('a') and key <= ord('z'):
                label = chr(key).upper()
                cv2.waitKey(2000)  # Wait for 2 second to pose the hands
                starttime = int(time.time())

    cap.release()
    cv2.destroyAllWindows()


    # # Process each file in the landmarks_data directory
    # for filename in os.listdir(output_dir):
    #     filepath = os.path.join(output_dir, filename)
    #     if os.path.isfile(filepath) and filename.endswith('.json'):
    #         print(f"Processing file: {filename}")
    #         show_landmark_data(filepath)

if __name__ == "__main__":
    main()