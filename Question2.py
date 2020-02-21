# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 01:12:09 2019

@author: Mert
"""

import ntpath

from functions import create_model
from functions import get_image_paths
from functions import add_feature_vector
from functions import find_closest_image

# Main code
image_width, image_height=256, 256
input_size = (image_width, image_height)

train_dir = './dataset/train'
validate_dir = './dataset/validation'
test_dir = './dataset/test'

# Creating a model
classifier = create_model(input_size)

# Load the last weights of the model in the question_1
checkpoint_path = './checkpoints/Q1/cp-0050.ckpt'
classifier.load_weights(checkpoint_path)
        
# Creating a dictinorty to store feature vectors
dictionary = {} 

# Getting image paths
training_image_paths = get_image_paths(train_dir)
validation_image_paths = get_image_paths(validate_dir)
test_image_paths = get_image_paths(test_dir)

# Adding validation and training feature vectors to the dictionary
layer_name='dense_1'
for file in training_image_paths:
    add_feature_vector(file, dictionary, classifier, layer_name, input_size)

for file in validation_image_paths:
    add_feature_vector(file, dictionary, classifier, layer_name, input_size)


# Finding the closest images to test set
# collect them into true predicts[] and wrong predicts[] according to its categories
true_predicts = []
wrong_predicts = []
for path in test_image_paths:
    result = find_closest_image(path, dictionary, classifier, layer_name, input_size)
    test_image_name=ntpath.basename(path)
    
    # Getting the categories of result and query image from its names
    result_category = result.split("_")[0]
    test_image_category = test_image_name.split("_")[0]
    
    msg = "{}->{}".format(test_image_name, result)
    if(result_category==test_image_category):        
        true_predicts.append(msg)
    else:
        wrong_predicts.append(msg)
        
# Print true predicts
print("\nPrinting true predicts...")
for msg in true_predicts:
     print(msg)
     
     
# Print wrong predicts
print("\nPrinting wrong predicts...")
for msg in wrong_predicts:
     print(msg)     
        
        
num_test_samples=len(test_image_paths)
num_true_predict=len(true_predicts)        
message="total: {}, true_predict: {}, accuracy: {}".format(num_test_samples, num_true_predict, num_true_predict/num_test_samples)
print(message)

