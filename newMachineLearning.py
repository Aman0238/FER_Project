from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


            ## define some variables
classTypes = 7
imgRow, imgColum = 48, 48
batch_size = 8

            ## call paths defined
theTrainingDataSet = '/Users/lenovo/PycharmProjects/my-python-program-one/Archive/train'
theTestDataSet = '/Users/lenovo/PycharmProjects/my-python-program-one/Archive/test'

            ## data augmentations generator
trainDataGenerator = ImageDataGenerator(
					rescale= 1./255,
					rotation_range= 30,
					shear_range= 0.3,
					zoom_range= 0.3,
					width_shift_range= 0.4,
					height_shift_range= 0.4,
					horizontal_flip= True,
					fill_mode= 'nearest')

            ## data rescaling or normalization
testDataGenerator = ImageDataGenerator(rescale=1. / 255)

            ## define parameters of training and validation generators
trainingGenerator = trainDataGenerator.flow_from_directory(
					theTrainingDataSet,
					color_mode='grayscale',
					target_size=(imgRow, imgColum),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

testGenerator = testDataGenerator.flow_from_directory(
							theTestDataSet,
							color_mode='grayscale',
							target_size=(imgRow, imgColum),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

            ## build the Convolutional Neural Network (CNN) Model
theModel = Sequential()

            ## the First Block

theModel.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(imgRow, imgColum, 1)))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(imgRow, imgColum, 1)))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(MaxPooling2D(pool_size=(2, 2)))
theModel.add(Dropout(0.2))

            ## the 2nd Block

theModel.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(MaxPooling2D(pool_size=(2, 2)))
theModel.add(Dropout(0.2))

            ## the 3rd Block

theModel.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(MaxPooling2D(pool_size=(2, 2)))
theModel.add(Dropout(0.2))

            ## the 4th Block

theModel.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(MaxPooling2D(pool_size=(2, 2)))
theModel.add(Dropout(0.2))

            ## the 5th Block

theModel.add(Flatten())
theModel.add(Dense(64, kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(Dropout(0.5))

            ## the 6th Block

theModel.add(Dense(64, kernel_initializer='he_normal'))
theModel.add(Activation('elu'))
theModel.add(BatchNormalization())
theModel.add(Dropout(0.5))

            ## the 7th Block

theModel.add(Dense(classTypes, kernel_initializer='he_normal'))
theModel.add(Activation('softmax'))

            ## show the overall built model summary
print(theModel.summary())

            ## set up some control flow method of the machine learning model
theCheckPoint = ModelCheckpoint('New_ML_vgg.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
theEarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
thePaceReduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)

theCallBackPoint = [theEarlyStop, theCheckPoint, thePaceReduction]

            ## model format type
theModel.compile(loss='categorical_crossentropy', optimizer ="adam", metrics=['accuracy'])

            ## define valuable variables
nb_train_samples = 28709
nb_validation_samples = 7178
epoch= 15

machineLearningPhase= theModel.fit_generator(trainingGenerator, steps_per_epoch=nb_train_samples // batch_size, epochs= epoch,
                                             callbacks= theCallBackPoint,
                                             validation_data= testGenerator,
                                             validation_steps= nb_validation_samples//batch_size)