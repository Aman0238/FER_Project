import tensorflow as tf
import os
import cv2
import random
from tensorflow import keras
from keras import layers
from faceDetect import *
from cropImage import *

img_array = cv2.imread("Train/0/Training_3908.jpg")
data_dir = "Train"
classifications = ["0","1","2","3","4","5","6"]
#classifications = ["0"] #for test purpose


for catagories in classifications:
    path = os.path.join(data_dir, catagories)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        #plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        #plt.show()
        break
    break

img_size = 224
newimg_array = cv2.resize(img_array, (img_size,img_size))
#plt.imshow(cv2.cvtColor(newimg_array,cv2.COLOR_BGR2RGB))
#plt.show()

trainable_data = []

def make_train_data():
    for catagories in classifications:
        path = os.path.join(data_dir, catagories)
        class_num = classifications.index(catagories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                newimg_array = cv2.resize(img_array, (img_size, img_size))
                trainable_data.append([newimg_array, class_num])
            except Exception as e:
                pass

make_train_data()
print(len(trainable_data))

##///////////////
tp= np.array(trainable_data)
#print('lalala' , tp.shape)

##///////////////

            #randomize the datas for a better unsequenced machine learning
random.shuffle(trainable_data)

X_feat = []
Y_lab = []
for feature, label in trainable_data:
    X_feat.append(feature)
    Y_lab.append(label)

X_feat = np.array(X_feat).reshape(-1,img_size, img_size, 3)
#print('lulu',type(X_feat))
#print('lili',X_feat.shape)

                #Normalize by below or by scikit(later)
X_feat = (X_feat/255.0)
#print('lele',X_feat[2])
Y = np.array(Y_lab)
#print('lll',Y[12])

            #DEEP learning moder for training - which is transfer learning

model = tf.keras.applications.MobileNetV2()  ##our pre-trained model
print('see the summary',model.summary())

            #Transfer learning - Tuning, while the weights will start from the last check point
base_input = model.layers[0].input  ##my input
base_output = model.layers[-2].output

#print('watch the base output',base_output)

final_output = layers.Dense(128)(base_output)  ##for the sake of adding new layer, after the global one
final_ouput = layers.Activation('relu')(final_output)  ##for the activation purpose
final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_ouput)  ##for the catagories available

print('the final output is' ,final_output)

new_model = keras.Model(inputs = base_input, outputs= final_output)  ##creating my new model
print('new summary is', new_model.summary())


# new_model.compile(loss="sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])
# new_model.fit(X_feat,Y_lab, epochs = 15)  ##train a model
# new_model.save('Final_model_64p35.h5')   ##save the model
# new_model = tf.keras.models.load_model('Final_model_64p35.h5')  ##load pretrained model
# new_model.evaluate  ## test data for its accuracy



##////////////////////////////////////////////////
new_model.compile(loss="sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])
new_model.fit(X_feat,Y, epochs = 10)  ##train a model
new_model.save('Mobile_Net_vgg.h5')   ##save the model
new_model = tf.keras.models.load_model('Mobile_Net_vgg.h5')  ##load pretrained model
new_model.evaluate  ## test data for its accuracy
##////////////////////////////////////////////////


img= cv2.imread('RECENT PHOTO.jpg')
print('hihi',img.shape)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()



faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)
face_roi = 0
for x, y, w, h in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    faces_s = faceCascade.detectMultiScale(roi_gray)
    if len(faces_s) == 0:
        print("face not found")
    else:
        for (ex, ey, ew, eh) in faces_s:
            face_roi = roi_color[ey: ey + eh, ex: ex + ew]
plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
plt.show()
final_image = cv2.resize(face_roi, (224,224))
final_image = np.expand_dims(final_image, axis =0)
final_image = final_image/255.0
#returned = cropImage(img)
Predictions = new_model.predict(final_image)

#pd = np.argmax(Predictions)
print('the preiddi' ,Predictions[0])
print('last value' , np.argmax(Predictions))
textPrint(img,np.argmax(Predictions))


# img1 = faceDetect(img,crop=True)
# cv2.imshow('returned from facedetect res', img1)
# print('cropped images shape', img1.shape)
#
#
# Predictions = new_model.predict(img1)
# #pd = np.argmax(Predictions)
# print(Predictions[0])
# print(np.argmax(Predictions))
#
# #textPrint(img1, pd)
# #cv2.waitKey()





