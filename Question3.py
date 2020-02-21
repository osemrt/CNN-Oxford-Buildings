# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 01:52:06 2019

@author: Mert
"""

import os
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

from functions import get_num_files
from functions import get_num_subfolders
from functions import create_img_generator_for_VGG16
from functions import plot_accuracy
from functions import plot_loss

# Suppress warning and informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

# Main code
image_width, image_height=256, 256
Training_Epochs=5
Batch_Size=32
Number_FC_Neurons=512

train_dir = './dataset/train'
validate_dir = './dataset/validation'
test_dir = './dataset/test'

num_epoch = Training_Epochs
batch_size = Batch_Size

num_classes = get_num_subfolders(train_dir)

num_train_samples = get_num_files(train_dir)
num_validate_samples = get_num_files(validate_dir)
num_test_samples = get_num_files(test_dir)


# Defining image generators for training and testing
train_image_gen = create_img_generator_for_VGG16()
validation_image_gen = create_img_generator_for_VGG16()

# Training image generator
train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(image_width, image_width),
    batch_size=batch_size,
    seed=42 # set seed for reproducability
    )

# Validation image generator
validation_generator = validation_image_gen.flow_from_directory(
    validate_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    seed=42 # set seed for reproducability
    )

# Test image generator
test_generator = ImageDataGenerator().flow_from_directory(
    test_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    seed=42 # set seed for reproducability
    )

# Load the VGG16 model and load it with it's pre-trained weights.
VGG16_base_model = VGG16(weights='imagenet', include_top=False) #include_top=False excludes final FC layer

print('VGG16 base model without last FC loaded')

# Freeze all layers
for layer in VGG16_base_model.layers:
    layer.trainable=False

print(VGG16_base_model.summary())

# Define the layers in the new classification prediction
x = VGG16_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation='relu')(x)      # new FC layer, random init
predictions = Dense(num_classes, activation='softmax')(x) # new softmax layer


# Define trainable model which links input 
# from the VGG16 base model to the new classification prediction layer
model = Model(inputs=VGG16_base_model.input, outputs=predictions)

# print model structure diagram
print(model.summary())

# Transfer Learning
print("\nPerforming Transfer Learning")
    
    
# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    

# Fit the Transfer Learning model to the data from the generators
history = model.fit_generator(
    train_generator,
    epochs=num_epoch,
    steps_per_epoch=num_train_samples//batch_size,
    validation_data=validation_generator,
    validation_steps=num_validate_samples//batch_size,
    class_weight='auto',
    shuffle=True)

# Plot training and validation accuracy
plot_accuracy(history)

# Plot training and validation loss
plot_loss(history)

# Print test set accuracy and loss values
scores = model.evaluate_generator(test_generator, num_test_samples/batch_size)
print("loss: {}, accuracy: {}".format(scores[0], scores[1]))