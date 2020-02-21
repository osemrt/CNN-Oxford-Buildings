# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:57:38 2019

@author: Mert
"""

import os
import sys
import ntpath
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.spatial import distance
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


# Plot data to see relationship in training and validation data
def plot_accuracy(hist):
    epoch_list=list(range(1, len(hist.history['accuracy']) + 1)) # values for x axis [1, 2, 3, 4, ..., # of epochs]
    plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
    plt.legend(('Training Accuracy', 'Validation Accuracy'))
    plt.show()
    return 0

# Plot data to see relationship in training and validation data
def plot_loss(hist):
    epoch_list=list(range(1, len(hist.history['loss']) + 1)) # values for x axis [1, 2, 3, 4, ..., # of epochs]
    plt.plot(epoch_list, hist.history['loss'], epoch_list, hist.history['val_loss'])
    plt.legend(('Training Loss', 'Validation Loss'))
    plt.show()
    return 0

# Get count of number of files in this folder and all subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

# Get count of number of subfolders directly below the folder in path
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

# Define image generators that will varitions of image with image rotated slightly, shifted up, down, left or right ...
def create_img_generator():
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

# Define image generators that will varitions of image with image rotated slightly, shifted up, down, left or right ...
def create_img_generator_for_VGG16():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

# Create a callback that saves the model's weights
def create_callback(checkpoint_path, period):    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=False,
    period=period)
    return cp_callback

# Load the previously saved weights
def load_last_weigts(checkpoint_dir, model):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model

# A function to create a model that has the following attributes
def create_model(input_size):
    # Initialising the CNN
    model = Sequential()
 
    # Convolution
    model.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu',
                     name='conv_0'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           name='pool_0'))
 
    # Adding a second convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu',
                     name='conv_1'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           name='pool_1'))
 
    # Adding a third convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu',
              name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           name='pool_2'))
 
    # Step 3 - Flattening
    model.add(Flatten(name='flate_0'))
 
    # Step 4 - Full connection
    model.add(Dense(units=1024, activation='relu',
                    name='dense_0'))
    
    model.add(Dense(units=1024, activation='relu',
                    name='dense_1'))
    #classifier.add(Dropout(0.5))
    model.add(Dense(units=17, activation='softmax',
                    name='output_layer'))
    
    return model

# A function to get all image paths
# in a specific  folder
def get_image_paths(root_path):
    files = []
    supported_extensions = ".bmp" ".pbm" ".pgm" ".ppm" ".jpeg" ".jpg" ".jpe" ".png" ".tiff" ".tif"
    # r=root, d=directories, f = files
    for r, d, f in os.walk(root_path):
        for file in f:
            extension = file.format().split('.')[-1]
            if extension in supported_extensions:
                files.append(os.path.join(r, file))
    return files


# A function to get a feature vector of the given image
def get_feature_vector(model, layer_name, img):
  
  layer_output=model.get_layer(layer_name).output 
  intermediate_model=Model(inputs=model.input, outputs=layer_output) 
  intermediate_prediction=intermediate_model.predict(img)

  return intermediate_prediction

# A function to visualize any layer after giving a image to the network
def visualize_conv_layer(model, layer_name, img):  
  layer_output = model.get_layer(layer_name).output 
  intermediate_model = Model(inputs = model.input, outputs = layer_output) 
  intermediate_prediction=intermediate_model.predict(img)
  
  row_size=4
  col_size=8
  
  img_index=0
 
  print(np.shape(intermediate_prediction))
  
  fig,ax=plt.subplots(row_size,col_size,figsize=(10,8))
 
  for row in range(0,row_size):
    for col in range(0,col_size):
      ax[row][col].imshow(intermediate_prediction[0, :, :, img_index], cmap='gray')
 
      img_index=img_index+1
      
# A function to prepare a image to prediction process     
def prepare_image(image_path, input_size):
    img = image.load_img(image_path, target_size=input_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# A function to add the feature of the given image to the dictionary
def add_feature_vector(image_path, dictionary, model, layer_name, input_size):    
    image_name=image_name=ntpath.basename(image_path)
    image=prepare_image(image_path, input_size)
    result=get_feature_vector(model, layer_name, image)
    feature_vector=result[0]    
    dictionary[image_name]=feature_vector  
    print("the feature vector of '{}' is added to the dictionary...".format(image_name))    

# A function to find the closest image to the training and validation images
def find_closest_image(image_path, dictionary, model, layer_name, input_size):
    min_dist=sys.float_info.max
    image=prepare_image(image_path, input_size)
    feature_vector=get_feature_vector(model, layer_name, image)
    feature_vectors=dictionary.values()
    for vector in feature_vectors:
        dst = distance.euclidean(feature_vector, vector)
        if dst < min_dist:
            min_dist=dst
            image_name=get_key(dictionary, vector)
    
    return image_name

# A function to get a key value from the give value in the dictionary
def get_key(dictionary, val): 
    for key, value in dictionary.items(): 
         if np.array_equal(val, value):
             return key 
  
    return "key doesn't exist"