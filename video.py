import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter

# Load model
model = joblib.load('models/gesture_model.pkl')

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (w, h))

def geometric_normalization(landmarks):
    x = np.array([lm.x for lm in landmarks])
    y = np.array([lm.y for lm in landmarks])
    z = np.array([lm.z for lm in landmarks])

    # Recenter by wrist (landmark 0)
    x = x - x[0]
    y = y - y[0]

    # Scale by Euclidean distance to middle fingertip (landmark 12)
    scale = np.sqrt(x[12]**2 + y[12]**2)
    if scale == 0:
        scale = 1.0
    x = x / scale
    y = y / scale

    # Interleave x, y, z to match training format: x1 y1 z1 x2 y2 z2 ...
    features = np.column_stack([x, y, z]).flatten()
    return features

# Stabilization window
window_size = 10
predictions_window = []

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_label = "No Hand"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw landmarks on frame
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Normalize and predict
        features = geometric_normalization(hand_landmarks.landmark)
        features = features.reshape(1, -1)
        pred = model.predict(features)[0]

        # Stabilization: take mode over window
        predictions_window.append(pred)
        if len(predictions_window) > window_size:
            predictions_window.pop(0)
        gesture_label = Counter(predictions_window).most_common(1)[0][0]

    # Display prediction on frame
    cv2.putText(frame, f'Gesture: {gesture_label}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show and save
    cv2.imshow('Gesture Recognition', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as output_video.mp4")