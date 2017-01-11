from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

def CNN(trainDir, validationDir, classNum):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, input_shape=(img_width, img_height, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(4, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer
    model.add(Convolution2D(8, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(16, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    # test
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zca_whitening=True,
            zoom_range=0.2,
            horizontal_flip=False)
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255, zca_whitening=True)
    train_generator = train_datagen.flow_from_directory(
            trainDir,
            target_size=(img_width, img_height),
            batch_size=32,
            color_mode='grayscale',
            class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
            validationDir,
            target_size=(img_width, img_height),
            batch_size=32,
            color_mode='grayscale',
            class_mode='categorical')
    model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)
    return model

if __name__ == '__main__':
    cropModel = CNN(train_data_dir, validation_data_dir, 2)
    cropModel.save_weights('cropWeights.h5')
    cropModel.save('cropModel.h5')
    classModel = CNN(train_class, validation_class, 25)
    classModel.save_weights('classWeights.h5')
    classModel.save('classModel.h5')