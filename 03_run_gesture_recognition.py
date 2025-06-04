import cv2
import mediapipe as mp
import numpy as np
import joblib

clf = joblib.load('models/svm_gesture_model.pkl')
encoder = joblib.load('models/label_encoder.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm = []
            for pt in handLms.landmark:
                lm.append(pt.x)
                lm.append(pt.y)
            lm_np = np.array(lm).reshape(1, -1)
            pred = clf.predict(lm_np)[0]
            label = encoder.inverse_transform([pred])[0]

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, f"{label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()