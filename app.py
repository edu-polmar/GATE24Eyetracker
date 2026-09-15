import cv2 as cv
import mediapipe as mp
import numpy as np

cam = cv.VideoCapture(0)

frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    out.write(frame)

    cv.imshow('Camera', frame)

    if cv.waitKey(1) == ord('q'):
        break

cam.release()
out.release()
cv.destroyAllWindows()
