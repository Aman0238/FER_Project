from cropImage import *
from keras.preprocessing.image import img_to_array
from keras.models import load_model

new_model = load_model(r'C:\Users\lenovo\PycharmProjects\my-python-program-one\New_ML_vgg.h5')

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

        ## set the rectangle background to white
rectangle_bgr = (255, 255, 255)
        ## make a black ground image
img = np.zeros ((500, 500))
        ## set some text
text = "to be printed text"
        ## capture the height and width of the box for the text
(tex_wi, tex_he) = cv2.getTextSize(text, font, 1.5, 1)[0]
        ## set the text origin
tex_offset_x = 10
tex_offset_y = img.shape[0] - 25
        ## make a small padding of two pixels of the box chords
box_ch = ((tex_offset_x, tex_offset_y), (tex_offset_x + tex_wi + 2, tex_offset_y - tex_he - 2))
cv2.rectangle(img, box_ch[0], box_ch[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (tex_offset_x, tex_offset_y), font, font_scale, (0, 0, 0), 1)

cap = cv2.VideoCapture(0)
        ## webcam open success checksum
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("webcam opening unsuccessful")

while True:
    ret, frame = cap.read()
    #faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(imgGray,1.3, 5)
    face_roi = 0

    for x, y, w, h in face:
        roi_gray = imgGray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48),interpolation=cv2.INTER_AREA)

        #roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # faces = faceCascade.detectMultiScale(roi_gray,1.3,5)
        # if len(faces) == 0:
        #     print("No Detected Faces")
        # else:
        #     for (ex, ey, ew, eh) in faces:
        #                    ## face cropping
        #         face_roi = roi_color[ey: ey + eh, ex: ex + ew]

        #new_model = tf.keras.models.load_model('New_ML_vgg.h5')

    #            #adding a 4th dimension
    #final_image = np.expand_dims(final_image, axis= 0)
    # final_image = final_image/255.0

        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        Predictions = new_model.predict(roi)[0]
        label_position = (x, y)

    #returned = cropImage(frame)

    #print(returned.shape)


    #font = cv2.FONT_HERSHEY_SIMPLEX
    # Prediction = deepface.DeepFace.analyze(final_image, actions=['emotion'])


    #Prediction = deepface.DeepFace.analyze(frame, actions=['emotion'])
    #Predictions= Prediction['dominant_emotion']

    #Predictions = new_model.predict(returned)

        font = cv2.FONT_HERSHEY_PLAIN

        if (np.argmax(Predictions) == 0):
            status = "ANGRY"

            x1, y1, w1, h1 = 0, 0, 175, 75
                    ## rectangle of black background
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    ## Text pudding
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int (h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Predictions) == 1):
           status = "DISGUST"

           x1, y1, w1, h1 = 0, 0, 175, 75
           ## rectangle of black background
           cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
           ## Text pudding
           cv2.putText(frame, status, (x1+ int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Predictions) == 2):
           status = "FEAR"

           x1, y1, w1, h1 = 0, 0, 175, 75
           ## rectangle of black background
           cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
           ## Text pudding
           cv2.putText(frame, status, (x1+ int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Predictions) == 3):
           status = "HAPPY"

           x1, y1, w1, h1 = 0, 0, 175, 75
           ## rectangle of black background
           cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
           ## Text pudding
           cv2.putText(frame, status, (x1+ int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Predictions) == 4):
           status = "NEUTRAL"

           x1, y1, w1, h1 = 0, 0, 175, 75
           ## rectangle of black background
           cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
           ## Text pudding
           cv2.putText(frame, status, (x1+ int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Predictions) == 5):
           status = "SAD"

           x1, y1, w1, h1 = 0, 0, 175, 75
           ## rectangle of black background
           cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
           ## Text pudding
           cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        else:
            status = "SURPRISE"

            x1, y1, w1, h1 = 0, 0, 175, 75
            ## rectangle of black background
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            ## Text pudding
            cv2.putText(frame, status, (x1+ int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

        cv2.imshow('LIVE WEB-CAM FEED EMOTION DETECTION', frame)
        # faceDetect(img, Prediction['dominant_emotion'])
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()