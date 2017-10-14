import os
import csv
import numpy as np
import cv2
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Conv2D,MaxPool2D,Dropout,Cropping2D
from keras.optimizers import Adam
import matplotlib.image as mpimg

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32,correction=0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #parse the names of the image
                name = './data/IMG/'+batch_sample[0].split('\\')[3]
                name_left = './data/IMG/'+batch_sample[1].split('\\')[3]
                name_right = './data/IMG/'+batch_sample[2].split('\\')[3]
                #extract values from images, augment the dataset
                center_image = mpimg.imread(name)
                left_image = mpimg.imread(name_left)
                right_image = mpimg.imread(name_right)
                center_image_flip = cv2.flip(center_image,1)
                left_image_flip = cv2.flip(left_image,1)
                right_image_flip = cv2.flip(right_image,1)
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                center_angle_flip = -center_angle
                left_angle_flip = center_angle_flip - correction
                right_angle_flip = center_angle_flip + correction
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                images.append(center_image_flip)
                angles.append(center_angle_flip)
                images.append(left_image_flip)
                angles.append(left_angle_flip)
                images.append(right_image_flip)
                angles.append(right_angle_flip)
            #putting everything toghether
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def custommodel(model, dropout):
    #normalizing, reduce mean and cropping the image
    model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))
    model.add(Lambda(lambda x:x/255.0 - 0.5))
    #adding convolutional and maxpool layers
    model.add(Conv2D(12, ( 7, 7 ),strides=( 2, 2 ) ,activation = 'relu'))
    #model.add(Dropout(dropout))
    model.add(Conv2D(24, ( 5, 5 ),strides=( 2, 2 ) ,activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(48, ( 3, 3 ),strides=( 2, 2 ) ,activation = 'relu'))
    model.add(Dropout(dropout))
    #model.add(Conv2D(48, ( 3, 3 ),strides=( 1, 1 ) ,activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    #model.add(Dropout(dropout))
    #Flatten to 1d and Fully Connecte layers
    model.add(Flatten())
    model.add(Dense(2000, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = None))
    model.compile(loss='mse',optimizer=Adam(lr=0.001))
    return model

#setting params and call the generator
dropout = 0.5
batch_size = 32
correction = 0.15
train_generator = generator(train_samples, batch_size=batch_size,correction=correction)
validation_generator = generator(validation_samples, batch_size=batch_size,correction=correction)
#build and train the model
model = Sequential()
model = custommodel(model,dropout)
model.summary()
model.fit_generator(train_generator, samples_per_epoch= len(train_samples) / batch_size, 
                    validation_data=validation_generator, nb_val_samples= len(validation_samples)/batch_size, nb_epoch=4)
model.save('mdmodel.h5')
