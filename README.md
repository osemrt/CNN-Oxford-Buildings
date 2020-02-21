
# CNN-Oxford-Buildings

All my current solutions are subject to Tensorflow. All assignment should be completed and passed the test set. If there is any errors or typo (or better solution!!!) please inform me! Click [here ](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)to get information about dataset.

## Question#1 - Classification
Using the Train and Validation set, design and classify a network with 3 convolutional hidden layers, followed by 2 fully-connected. Show your convolutional neural network configuration on the test set as accuracy and loss.

## Question#2 - Feature Vector
**Using your completed training network;**

**a)**  Save the outputs of each image in the dataset on the fully-connected layer (it can be the first or the second) as a feature vector for that picture.

**b)**  Calculate the closest pictures for the images in the test set that you previously reserved using this feature vector. At this stage, you can use Euclidean Distance for distance calculation.

## Question#3 - Transfer Learning

In the first two questions, you were expected to design a new network, train it, and obtain an feature vector for each image based on the weights of the trained network. In this question, you are expected to download the weights of the trained VGG-16 network with the built-in functions of Keras in the Imagenet contest and to get the closest images for the images in your test set using the weights in the first fully-connected layer of the VGG-16 network. 

