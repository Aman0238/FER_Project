import cv2
import tensorflow as tf
from cropImage import *
from videoFaceDetectModule import *
from keras.preprocessing.image import img_to_array

            ## Load some pretrained models for the prediction and define the classes
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
theClassifier = tf.keras.models.load_model('New_ML_vgg.h5')
theClasses = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            ## define some parameters for video capturing
cap = cv2.VideoCapture(0)
prevTime = 0
detect = FaceDetector()
            ## start the loop for the prediction
while True:
            ## capture single frame of video
    ret, frame = cap.read()
    img, theBox = detect.findFaces(frame, True)  # call the videoFaceDetectModule

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the bgr img to gray for better prediction
    theFaces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # return list of rectangles after detection of d/t input size

            ## loop only in the region of interest which is the face
    for (x,y,w,h) in theFaces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        regofInterest_Gray = imgGray[y:y + h, x:x + w]
        regofInterest_Gray = cv2.resize(regofInterest_Gray, (48, 48), interpolation=cv2.INTER_AREA)  # resize the img

        if np.sum([regofInterest_Gray])!= 0:  # if a face is detected
            regionOfInterest = regofInterest_Gray.astype('float') / 255.0  # normalizing
            regionOfInterest = img_to_array(regionOfInterest)  # convert from PIL Image instance to a Numpy array
            regionOfInterest = np.expand_dims(regionOfInterest, axis=0)  # expand the shape of the array

                ## make a prediction on the region of interest using the pretrained model, then lookup in the classes
            prediction = theClassifier.predict(regionOfInterest)[0]  # make a prediction
            aTag= theClasses[prediction.argmax()]  # get a tag of response form the class defined with max value of prediction
            cv2.putText(img, aTag, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # put the prediction
        else:
            cv2.putText(img,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            ## caputre the frame per second of the motion picture
    currentTime = time.time()
    framePerSecond = 1 / (currentTime - prevTime)
    prevTime = currentTime
    cv2.putText(img, f'FPS: {int(framePerSecond)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
    cv2.imshow('show', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # click q to quit the program
        break

cap.release()
cv2.destroyAllWindows()







