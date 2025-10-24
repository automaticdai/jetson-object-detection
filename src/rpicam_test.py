import cv2 as cv
import numpy as np


cap = cv.VideoCapture(-1)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv.imshow("Camera", frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
