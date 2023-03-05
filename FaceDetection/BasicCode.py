import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_facedetection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_facedetection.FaceDetection(min_detection_confidence=0.5)

cTime = 0
pTime = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame")
        continue
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False

    # Our webCamp image/frame is in BGR formate, First we convert our
    # image from BGR to RGB b/z mediapipe only proccess rgb images
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgbImage)
    frame.flags.writeable = True
    print(result.detections)
    if result.detections:
        for id, detection in enumerate(result.detections):
            # mp_drawing.draw_detection(frame, detectin)
            # print(id, detectioqqn)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxc = detection.location_data.relative_bounding_box
            image_height, image_width , image_channel = frame.shape
            bbox = int(bboxc.xmin * image_width), int(bboxc.ymin * image_height), \
                   int(bboxc.width * image_width), int(bboxc.height * image_height)
            cv2.rectangle(frame, bbox , (255,0,255),2)
            cv2.putText(frame,f'{int(detection.score[0]*100)}%',
                        (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1)
            # cx,cy = int(bboxc.xmin * image_width), int(bboxc.ymin * image_height)
            # cv2.circle(frame, (cx,cy),(255,255,0),cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
