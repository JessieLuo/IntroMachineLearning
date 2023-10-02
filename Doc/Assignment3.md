# Assignment 3: PCA, K-means and Kernel Methods

## Description

The above data is a sub-sampled version of the MNIST dataset. It contains images for 10 digits (10 classes). The dataset contains 6,000 samples. The images from the data set have the size 28 x 28. They are saved in the cv data files. Every line of these files consists of an image, i.e. 785 numbers between 0 and 1. The first number of each line is the label, i.e. the digit which is depicted in the image. The following 784 numbers are the pixels of the 28 × 28 image.

## Question 1

Perform PCA on the dataset to reduce each sample into a 10-dimensional feature vector.

Show the covariance matrix of the transformed data.

Please also copy your code snippet here.

## Question 2

Perform k-means clustering to cluster the dataset (without applying PCA) into 10 groups.

## Question 3

Please plot the loss curve, that is, the change of loss value of the k-means algorithm with respect to the number of iterations

## Question 4

Please use the first 4000 samples as the training set and remaining 2000 samples as the validation set, and design a way to choose the best k in k-means algorithm.

## Question 5

Please implement kernel k-means algorithm with RBF-kernel, that is,

$k(X_i,X_j)=exp(\frac{-||x_i-x_j||^2_2}{2\sigma^2})$.

The hyper-parameter can be empirically set to

$2\sigma^2=\frac{1}{N^2}\sum_{i=1}^N\sum_{j=1}^N||x_i-x_j||^2_2$

Please only use the first 500 samples and cluster them into 5 groups. This is for reducing the running time of your code.

TIPS: If you can use matrix operations to replace summations, your code will be more efficient. However, this is just optional.
