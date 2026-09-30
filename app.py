import cv2 as cv
import mediapipe as mp
import numpy as np

print(mp.__file__)
print(hasattr(mp, 'solutions'))

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
    )

VANSTER_IRIS = [474, 475, 476, 477]
HOGER_IRIS = [469, 470, 471, 472]

# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]


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
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w = frame.shape[:2]

    if results.multi_face_landmarks:
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        cv.polylines(frame, [mesh_points[VANSTER_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
        cv.polylines(frame, [mesh_points[HOGER_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[VANSTER_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[HOGER_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv.circle(frame, center_left, int(l_radius), (255,0,255), 2, cv.LINE_AA)
        cv.circle(frame, center_right, int(r_radius), (255,0,255), 2, cv.LINE_AA)

    out.write(frame)
    cv.imshow('Iris-detektion', frame)

    if cv.waitKey(

        1) & 0xFF == ord('q'):
        break

cam.release()
out.release()
cv.destroyAllWindows()
