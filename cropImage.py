import cv2
from keras.preprocessing.image import img_to_array

import numpy as np
import matplotlib.pyplot as plt

def cropImage(img):
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    face_roi = 0
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faces_s = faceCascade.detectMultiScale(roi_gray)
        if len(faces_s) == 0:
            print("face not found")
        else:
            for (ex, ey, ew, eh) in faces_s:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew]

    # plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    # plt.show()
    final_image = cv2.resize(face_roi, (224,224),interpolation=cv2.INTER_AREA)
    print(final_image.shape)

    # final_image = np.expand_dims(final_image, axis =0)
    # final_image = final_image/255.0

    final_image = roi_gray.astype('float') / 255.0
    final_image = img_to_array(final_image)
    roi = np.expand_dims(final_image, axis=0)
    # print(final_image.shape)

    return final_image
