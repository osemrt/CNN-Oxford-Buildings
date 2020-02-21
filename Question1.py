# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 01:52:06 2019

@author: Mert
"""
import os
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator

from functions import create_model
from functions import get_num_files
from functions import get_num_subfolders
from functions import create_callback
from functions import load_last_weigts
from functions import plot_accuracy
from functions import plot_loss
from functions import create_img_generator

# Suppress warning and informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

# Main code
image_width, image_height=256, 256
Training_Epochs=100
Batch_Size=8
input_size = (image_width, image_height)

train_dir = './dataset/train'
validate_dir = './dataset/validation'
test_dir = './dataset/test'

num_classes = get_num_subfolders(train_dir)

num_train_samples = get_num_files(train_dir)
num_validate_samples = get_num_files(validate_dir)
num_test_samples = get_num_files(test_dir)

num_epoch = Training_Epochs
batch_size = Batch_Size

# Define data pre-processing
# Defining image generators for training and testing
train_image_gen = create_img_generator()
validation_image_gen = create_img_generator()

# Connect the image generator to a folder contains the source images the image generator alters
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


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "checkpoints/q1_V5/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

classifier=create_model(input_size)
 
# print model structure diagram
print(classifier.summary())

# Compiling the CNN
classifier.compile(adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

cp_callback = create_callback(checkpoint_path, 1)

hist = classifier.fit_generator(
    train_generator,
    epochs=num_epoch,
    steps_per_epoch=num_train_samples//batch_size,
    validation_data=validation_generator,
    validation_steps=num_validate_samples//batch_size,
    class_weight='auto',
    callbacks=[cp_callback]
    )

# Plot training and validation accuracy
plot_accuracy(hist)

# Plot training and validation loss
plot_loss(hist)

# Print test set accuracy and loss values
scores = classifier.evaluate_generator(test_generator, num_test_samples/batch_size)
print("loss: {}, accuracy: {}".format(scores[0], scores[1]))


