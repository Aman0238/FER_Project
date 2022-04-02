import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionCon= 0.5):

        self.minDetectionCon = minDetectionCon

            ## Using medipipe library for face detection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)

    def findFaces(self,img, draw= True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bound_boxes =[]

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # self.mpDraw.draw_detection(img, detection)
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bound_box_class = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bound_box_class.xmin * iw), int(bound_box_class.ymin * ih), \
                       int(bound_box_class.width * iw), int(bound_box_class.height * ih)
                bound_boxes.append([id,bbox,detection.score])
                # print(bbox)
                # print(img.shape)
                if draw == True:
                    img = self.cornerPaint(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] *100)}%',\
                            (bbox[0] + bbox [2], bbox[1] + bbox[3]), cv2.FONT_HERSHEY_PLAIN,\
                            3, (0, 255, 0), 2)

        return img, bound_boxes

    def cornerPaint(self, img, bbox, len=25, thick= 5, recThick= 2):
        x, y, w, h = bbox
        x1 = x + w
        y1 = y + h

        cv2.rectangle(img, bbox, (255, 0, 0), recThick)
                ## Corner POS= TOP left
        cv2.line(img, (x, y), (x + len, y), (255, 0, 255), thick)
        cv2.line(img, (x, y), (x , y + len), (255, 0, 255), thick)
                ## Corner POS= Bottom Right
        cv2.line(img, (x1, y1), (x1 - len, y1), (255, 0, 255), thick)
        cv2.line(img, (x1, y1), (x1 , y1 - len), (255, 0, 255), thick)
                ## Corner POS= Top Right
        cv2.line(img, (x1, y), (x1 - len, y), (255, 0, 255), thick)
        cv2.line(img, (x1, y), (x1, y + len), (255, 0, 255), thick)
                ## Corner POS= Bottom left
        cv2.line(img, (x, y1), (x + len, y1), (255, 0, 255), thick)
        cv2.line(img, (x, y1), (x, y1 - len), (255, 0, 255), thick)

        return img
def main():
    cap = cv2.VideoCapture("Woman.mp4")
    pTime = 0
    detection = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detection.findFaces(img,True)
        #print(img.shape)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
        cv2.imshow('show', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

        cv2.waitKey(1)

if __name__ == "__main__":
    main()