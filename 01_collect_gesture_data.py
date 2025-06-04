import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

features = []
gesture_name = input("Enter gesture label (e.g., 'peace', 'fist'): ")

cap = cv2.VideoCapture(0)
print("Press 'q' to stop recording...")

while True:
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm = []
            for pt in handLms.landmark:
                lm.append(pt.x)
                lm.append(pt.y)
            features.append(lm)

    cv2.imshow("Collecting Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

np.save(f'data/landmarks/{gesture_name}_features.npy', features)
print(f"Saved {len(features)} samples for gesture: {gesture_name}")

cap.release()
cv2.destroyAllWindows()