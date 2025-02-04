import cv2
import numpy as np
import os
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
VIDEO_PATH = os.getenv("VIDEO_PATH")

# Load the trained MLP model
model = load_model("C:/Users/juaki/OneDrive/Escritorio/ASLHandRecognitionModel/best_model.keras")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Labels (A-Z)
labels = sorted(os.listdir("C:/Users/juaki/OneDrive/Escritorio/ASLHandRecognitionModel/asl_dataset_keypoints"))

# Open video file or webcam
cap = cv2.VideoCapture("C:/Users/juaki/OneDrive/Escritorio/ASLHandRecognitionModel/Prueba1.mp4")

# Store predictions to form words
predicted_sequence = deque(maxlen=10)  # Keeps last 10 predictions to form words

frame_count = 0  # Counter for frame skipping
skip_frames = 5  # Number of frames to skip between predictions
last_prediction = None  # Store last stable prediction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only process every 'skip_frames' frames
    if frame_count % skip_frames == 0:
        # Convert frame to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z])

                keypoints = np.array(keypoints).flatten().reshape(1, -1)

                # Predict the letter
                prediction = model.predict(keypoints)
                predicted_letter = labels[np.argmax(prediction)]

                # Only update the sequence if the prediction is stable
                if predicted_letter != last_prediction:
                    last_prediction = predicted_letter
                    predicted_sequence.append(predicted_letter)

    # Display last stable prediction
    if last_prediction:
        cv2.putText(frame, f"Letter: {last_prediction}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video output
    cv2.imshow("ASL Letter Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print the detected sequence as a word
final_word = "".join(predicted_sequence)
print("Predicted Word:", final_word)
