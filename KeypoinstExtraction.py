import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

DATASET_PATH = "C:/Users/juaki/OneDrive/Escritorio/ASLHandRecognition/asl_dataset_raw"
OUTPUT_PATH = "C:/Users/juaki/OneDrive/Escritorio/ASLHandRecognition/asl_dataset_keypoints"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Loop through each letter folder
for letter in os.listdir(DATASET_PATH):
    letter_path = os.path.join(DATASET_PATH, letter)
    output_letter_path = os.path.join(OUTPUT_PATH, letter)

    if not os.path.exists(output_letter_path):
        os.makedirs(output_letter_path)

    for image_name in os.listdir(letter_path):
        image_path = os.path.join(letter_path, image_name)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z])

                keypoints = np.array(keypoints).flatten()

                # Save keypoints
                np.save(os.path.join(output_letter_path, image_name.replace('.jpeg', '.npy')), keypoints)

print("Keypoints extraction complete")
