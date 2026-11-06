import sys
import time
import csv

import cv2 as cv
import mediapipe as mp
import numpy as np

#version control
print(mp.__file__)
print(hasattr(mp, 'solutions'))
print(mp.__version__)


# ------ SETTINGS ------
Participant = 'nikolas' #output video file name
Text_language = 'svenska'
Folder = 'data'
FPS = 20.0
LEFT_EYE = {'iris': [474, 475, 476, 477], 'corner': (33, 133), 'lid': (159, 145)}
RIGHT_EYE = {'iris': [469, 470, 471, 472], 'corner': (362,263), 'lid': (386, 374)}
COLOR_EYE = (255, 0, 255) #purple


#------ Mediapipe Face Mesh Setup ------

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

def relative_pos(mesh_points, eye):
    (cx, cy), radius = cv.minEnclosingCircle(mesh_points[eye['iris']])
    x_a, x_b = mesh_points[eye['corner'][0]][0], mesh_points[eye['corner'][1]][0]
    y_up, y_down = mesh_points[eye['lid'][0]][1], mesh_points[eye['lid'][1]][1]
    rel_x = (cx-x_a) / (x_b - x_a)
    rel_y = (cy - y_up) / (y_down - y_up)
    return rel_x, rel_y, cx, cy, radius

def main():
    t0 = time.perf_counter()
    frame_nr = 0
    name = f'{Participant}_{Text_language}'

    cam = cv.VideoCapture(find_camera_index(), cv.CAP_DSHOW)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(f'{Folder}/{name}.mp4', fourcc, FPS, (640, 480))

    csv_file = open(f'{Folder}/{name}.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['frame', 't', 'rel_x', 'rel_y', 'right_x', 'right_y', 'left_x', 'left_y'])

    if not cam.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    
    while True:
        ret, frame = read_video(cam)
        if not ret:
            print("Failed to grab frame")
            break

        out.write(frame)
        t = time.perf_counter() - t0
        h, w = frame.shape[:2]
        results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mesh_points = np.array([[p.x * w, p.y * h] for p in landmarks]).astype(int)

            hx, hy, hcx, hcy, hr = relative_pos(mesh_points, RIGHT_EYE)
            vx, vy, vcx, vcy, vr = relative_pos(mesh_points, LEFT_EYE)

            writer.writerow([frame_nr, round(t, 4), (hx + vx) / 2, (hy + vy) / 2, hx, hy, vx, vy])

            cv.circle(frame, (int(hcx), int(hcy)), int(hr), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, (int(vcx), int(vcy)), int(vr), (255, 0, 255), 2, cv.LINE_AA)

        else:
            writer.writerow([frame_nr, round(t, 4)] + [float('nan')] * 6)


        frame_nr += 1
        cv.imshow('Iris-detektion', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    out.release()
    csv_file.close()
    cv.destroyAllWindows()


# Running main function
if __name__ == "__main__":
    main()