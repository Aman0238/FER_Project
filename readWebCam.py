import cv2
from videoFaceDetectModule import *
import tensorflow as tf
from cropImage import *
from textPrint import *

# # frameWidth = 500
# # frameHeight = 400
# cap = cv2.VideoCapture(0)
# # cap.set(3, frameWidth)
# # cap.set(4, frameHeight)
# # cap.set(10, 150)
# while True:
#     success, img = cap.read()
#     FaceDetector.findFaces(img, True)
#     #cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cap = cv2.VideoCapture(0)
pTime = 0
detection = FaceDetector()

##///////////
# new_model = tf.keras.models.load_model('Final_model_94p69.h5')
##//////////////

while True:
    success, img = cap.read()
    img, bboxs = detection.findFaces(img,True)

    ##////////////
    # returned = cropImage(img)
    # Predictions = new_model.predict(returned)
    # textPrint(img, np.argmax(Predictions))
    ##////////////


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
    cv2.imshow('show', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(1)