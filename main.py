import cv2
import deepface.DeepFace
import numpy as np
import matplotlib as plt

from split_merge import splitImages
from stackImages import *
from faceDetect import *
from textPrint import *
from keras.models import load_model
from time import sleep
#from colorFinder import *
from keras.preprocessing.image import img_to_array



#img1= cv2.imread('pandi.jpg')
#r,g,b= splitImages(img1)
#colorFind(img1)
#imgStack = stackImages(0.5, ([img1,img1,img1], [r,g,b]))
#cv2.imshow("ImageStack", imgStack)
#cv2.waitKey()



face_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
#classifier = load_model('New_ML_vgg.h5')  ##load pretrained model
classifier =load_model(r'C:\Users\lenovo\PycharmProjects\my-python-program-one\New_ML_vgg.h5')
class_labels = ['Angry', 'Disgust','Fear','Happy','Neutral','Sad','Surprise']


#//////////////////////////////////////////////////
import deepface
img= cv2.imread('RECENT PHOTO.jpg')



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


    if np.sum([roi_gray])!=0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

    # make a prediction on the ROI, then lookup the class

        preds = classifier.predict(roi)[0]
        label=class_labels[preds.argmax()]
        label_position = (x,y)
        cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
        cv2.putText(img,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
#cv2.imshow('show', img)
#predictions = deepface.DeepFace.analyze(img, actions=['emotion'])
#print(predictions)
#faceDetect(img, predictions['dominant_emotion'])
#print(predictions['dominant_emotion'])
#textPrint(img,predictions['dominant_emotion'])

cv2.waitKey()