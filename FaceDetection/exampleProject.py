#Here I going to show you, how to use module that we create fro facedetection
#and you will see advantages of module that we don't
# need to write the code for facedetection again and again

import time

import cv2
import mediapipe as mp
import FaceDetectionModuel as fdm

def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = fdm.FaceDetector()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue
        bboxlist, image = detector.find_face(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()