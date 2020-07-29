import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) #skip the headings
    for line in reader:
        samples.append(line)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples,test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
                    for i in range(3):
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                        center_angle = float(batch_sample[3])
                        images.append(center_image)
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+correction)
                        elif(i==2):
                            angles.append(center_angle-correction)
                        
                        # Augmentation of data.
                     
                        # Flip function
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) 

# compile and train the model using the generator function            
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, Lambda, Cropping2D
#from keras.utils.vis_utils import plot_model

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) # Trimmed image          
model.add(Conv2D(24, 5, 5, activation="elu", subsample=(2,2)))
model.add(Conv2D(36, 5, 5, activation="elu", subsample=(2,2)))
model.add(Conv2D(48, 5, 5, activation="elu", subsample=(2,2)))
model.add(Conv2D(64, 3, 3, activation="elu"))
model.add(Conv2D(64, 3, 3, activation="elu"))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')
print('Done! Model saved!')
