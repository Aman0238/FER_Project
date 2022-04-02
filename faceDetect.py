import cv2
from textPrint import *


def faceDetect(img):
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('gray img shape'+ gray.shape)

    #img = cv2.imread('RECENT PHOTO.jpg')
    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)


