import cv2 as cv
import mediapipe as mp
import numpy as np

print(mp.__file__)
print(hasattr(mp, 'solutions'))

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
    )

VANSTER_IRIS = [474, 475, 476, 477]
HOGER_IRIS = [469, 470, 471, 472]

camera_index = None
for i in range(5):
    test_cam = cv.VideoCapture(i, cv.CAP_DSHOW)
    if test_cam.isOpened():
        print(f'Camera found on index {i}')
        if camera_index is None:
            camera_index = i
    test_cam.release()

if camera_index is None:
    print('ERROR: No camera found')

cam = cv.VideoCapture(camera_index, cv.CAP_DSHOW)

frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
print(f'Bredd: {frame_width}, Höjd: {frame_height}')

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
print('VideoWriter opened:', out.isOpened())

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        for idx in VANSTER_IRIS + HOGER_IRIS:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

    out.write(frame)
    cv.imshow('Iris-detektion', frame)

    if cv.waitKey(

        1) & 0xFF == ord('q'):
        break

cam.release()
out.release()
cv.destroyAllWindows()
