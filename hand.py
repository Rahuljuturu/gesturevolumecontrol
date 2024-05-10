import cv2
import mediapipe as mp
import numpy as np
import subprocess
import math

def set_volume(vol):
    # Calculate the volume as an integer percentage (0 to 100)
    volume_percent = max(0, min(100, int(vol)))
    # Use AppleScript to set the volume
    command = f"osascript -e 'set volume output volume {volume_percent}'"
    subprocess.run(command, shell=True) #for mac os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    # Convert the image color to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if len(lmList) > 8:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
                cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                length = math.hypot(x2 - x1, y2 - y1)

                # Interpolate the length to get volume between the volume range
                vol = np.interp(length, [50, 300], [0, 100])
                set_volume(vol)

                # Display volume bar on the screen (for visualization)
                volBar = np.interp(length, [50, 300], [400, 150])
                volPer = np.interp(length, [50, 300], [0, 100])
                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
