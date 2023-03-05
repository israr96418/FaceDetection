import time

import cv2
import mediapipe as mp



class FaceDetector():
    def __init__(self):
        self.mp_facedetection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_facedetection.FaceDetection(min_detection_confidence=0.5)

    def find_face(self, image, draw = True):
        bboxs_list = []
        image.flags.writeable = False
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.face_detection.process(rgbImage)
        image.flags.writeable = True
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxc = detection.location_data.relative_bounding_box
                image_height, image_width, image_channel = image.shape

                bbox = int(bboxc.xmin * image_width), int(bboxc.ymin * image_height), \
                       int(bboxc.width * image_width), int(bboxc.height * image_height)
                bboxs_list.append([id, bbox, detection.score])
                if draw:
                    image = self.fancy_detection(image, bbox)
                    cv2.putText(image, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)


        return bboxs_list,image

    def fancy_detection(self, image, bbox, l=30, thickness = 2, rectangle_thickness=2):
        x, y, w, h = bbox
        x1, y1, = x + w, y + h
        cv2.rectangle(image,bbox,(255,255,0),rectangle_thickness)

        # Top left corner
        cv2.line(image,(x, y),(x + l, y),(0,0,255) , thickness)
        cv2.line(image, (x, y), (x, y + l), (0, 0, 255), thickness)

        # Top right corner
        cv2.line(image, (x1, y), (x1 - l, y), (0, 0, 255), thickness)
        cv2.line(image, (x1, y), (x1, y + l), (0, 0, 255), thickness)

        # Bottom left corner
        cv2.line(image, (x, y1), (x + l, y1), (0, 0, 255), thickness)
        cv2.line(image, (x, y1), (x, y1 - l), (0, 0, 255), thickness)

        # Bottom right corner
        cv2.line(image, (x1, y1), (x1 - l, y1), (0, 0, 255), thickness)
        cv2.line(image, (x1, y1), (x1, y1 - l), (0, 0, 255), thickness)

        return image

def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = FaceDetector()
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