# Data-Science-Nanodegree_Project-4

![alt text](https://github.com/LamyaMK/Data-Science-Nanodegree_Project-4/blob/main/sample_dog_output.png?raw=true)
### Problem Introduction:
Welcome to the dog breed classifier project. This project uses Convolutional Neural Networks (CNNs)! In this project, I built a pipeline to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.
##### The Problem:
Given a an image, we need to determine whether the image contains a human, dog, or neither. Then,
-if a dog is detected in the image, return the predicted breed.
-if a human is detected in the image, return the resembling dog breed.
-if neither is detected in the image, provide output that indicates an error
##### Strategy to solve the problem: 

In order to solve the problem, I followed these stips:

Step 0: Import Datasets
Step 1: Detect Humans
Step 2: Detect Dogs
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
Step 6: Test Your Algorithm
Step 7: Create a web app to use the solution

##### Metric:
In this project, we aim to improve the model accuracy, we will chose the checkpoint with the highest accuracy.
Acuuracy metric is defined as the number of correct predictions over the number of total predictions.

#### Requirements:
To run the experiment, you need to install jupyter notebook to run dog_app.ipynb
Python requirements:
- Flask
- sklearn       
- keras
- numpy
- import
- cv2
- matplotlib
- tqdm
- PIL

Please download this [file](https://drive.google.com/file/d/1SL4zVqL_LZXXHyxjVhIoUxRmzxoIDmKU/view?usp=sharing), and move it bottleneck_features folder.

### EDA:
**Dog Dataset:**

There are 133 total dog categories.

There are 8351 total dog images.


There are 6680 training dog images.

There are 835 validation dog images.

There are 836 test dog images.

**Human Dataset:**

There are 13233 total human images.

**Human Face Detector:** 

I used OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.

**Dog Detector:**

I used a pre-trained ResNet-50 model to detect dogs in images.

##### Modelling:
1) My own CNN model:
```

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 224, 224, 16)      208       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 112, 112, 16)      0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 112, 112, 32)      2080      
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 56, 56, 64)        8256      
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 28, 28, 128)       32896     
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 14, 14, 128)       0         
_________________________________________________________________
global_average_pooling2d_2 ( (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               66048     
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 133)               68229     
=================================================================
```
2) VGG16 model
3) Resnet50 model

##### Hyperparameter tuning:

I didn't do much work on hyperparamters tuning; I've only set the number of epochs to 20, and used GlobalAveragePooling2D layer instead of flatten in the head.

### Results:

Accuracy of the CNN model trained from scratch to classify dog breed: 11.24%

Accuracy of the VGG16 model trained from imagenet weights to classify dog breed: 43.66%

Accuracy of the Resnet50 model trained from imagenet weights to classify dog breed: 79.67%

### Conclusion
In order to make my algorithm easy to use, I created a web app (app.py), that one can select any image and use it to test the model.

![alt text](https://github.com/LamyaMK/Data-Science-Nanodegree_Project-4/blob/main/images/snapshot.png?raw=true)

The project was fun! I enjoyed working on it, and the only diffeculty I faced is developing with the web app.

### Improvements:
The algorithm for sure is not 100% perfect. It can't always detect faces from the image, and sometime it gives wrong predicitons. The algorithm can be improved by the followings:


1) apply data augmentations.
2) tune the hyperparameters and the head of the model.
3) use deep ensemble (more than one model), which solves the problem of model uncertainty and help detecting outliers (images that don't belong to the data distribution)



