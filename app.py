import sys

import cv2 as cv
import mediapipe as mp
import numpy as np


print(mp.__file__)
print(hasattr(mp, 'solutions'))
print(mp.__version__)


# ------ SETTINGS ------

Video_file = 'HELLO.mp4'
FPS = 20.0
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
COLOR_IRIS = (255, 0, 255)


#test

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
    )



# ------ Functions ------
def find_camera_index():
    for i in range(5):
        test_cam = cv.VideoCapture(i, cv.CAP_DSHOW)
        works = test_cam.isOpened()
        test_cam.release()
        if works:
            print(f'Camera found at index {i}')
            return i
    return None

def read_video(cam):
    ok, frame = cam.read()
    if ok:
        frame = cv.flip(frame, 1)
    return ok, frame

def draw_iris(frame, mesh_points, iris_index):
    (cx, cy), radius = cv.minEnclosingCircle(mesh_points[iris_index])
    cv.circle(frame, (int(cx), int(cy)), int(radius), COLOR_IRIS, 2, cv.LINE_AA)

def main():
    cam = cv.VideoCapture(find_camera_index(), cv.CAP_DSHOW)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(Video_file, fourcc, FPS, (640, 480))
    if not cam.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    while True:
        ret, frame = read_video(cam)
        if not ret:
            print("Failed to grab frame")
            break

        h, w = frame.shape[:2]
        results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mesh_points = np.array([[p.x * w, p.y * h] for p in landmarks]).astype(int)

            draw_iris(frame, mesh_points, LEFT_IRIS)
            draw_iris(frame, mesh_points, RIGHT_IRIS)

        out.write(frame)
        cv.imshow('Iris-detektion', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()