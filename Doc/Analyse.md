# Analysis of Assignment3

## Overview

## Question 1

### Analysis Q1

- After preprocessing, Use PCA
- How to show n_component matrix(covariance matrix)?
  - Cal Mean of each sample(each row)
  - use $x_i-\bar{x}$ to get the centralized data
  - Calculate the **Covariance Matrix** of the centralized data.
  - Compute Eigenvalues and Eigenvectors, and descended order the eigenvalues in e-vector
  - choose the the first 'n_components' column and all rows of eigenvector to form the projection matrix
  - Use original feature to multiple the projection matrix to get the covariance matrix of the transformed data

## Question 2

### Analysis Q2

- KMeans Steps:
    1. Define $k$
    2. initiate the centroids
        - $k$ is the group of dataset?
    3. Calculate distance
    4. Assign centroids
    5. Update centroid location
    6. Repeat steps 3-5
- KMeans Standard Steps:
  1. Initialization
        - choose cluster num $k$
        - One of method choose $k$:
          - Randomly select $k$ data points (from the dataset) to be the initial centroids
  2. Assignment
        - Assign each data point to the nearest centroid
           - This forms $k$ clusters.
           - Euclidean distance usually used
  3. Update centroid
        - compute the mean of the data points in that cluster and set the centroid to that mean.
        - This mean becomes the new centroid for that cluster.
  4. Convergence Check
        - Check if the centroids have changed.
        - If they haven't (or the change is below a certain threshold), then the algorithm has converged, and you can stop.
        - If they have changed, go back to the "Assignment" step.(Repeat 2-3 until converge)

## Question 5

### Analysis Q5

kernel k-means algorithm with RBF-kernel, that is,

$k(X_i,X_j)=exp(\frac{-||x_i-x_j||^2_2}{2\sigma^2})$.

The hyper-parameter can be empirically set to

$2\sigma^2=\frac{1}{N^2}\sum_{i=1}^N\sum_{j=1}^N||x_i-x_j||^2_2$

$K(X_i,X_j)=exp(\frac{-||x_i-x_j||^2_2}{\frac{1}{N^2}\sum_{i=1}^N\sum_{j=1}^N||x_i-x_j||^2_2})$ ???

- Kernel
  - In machine learning, a kernel is a function that computes the dot product of the transformed vectors in a higher-dimensional space without explicitly computing the transformation.
  - It's a way to capture non-linear relationships in the data.
- RBF kernel
  - RBF stands for Radial Basis Function.
  - It's a popular kernel used in various machine learning algorithms, especially in support vector machines.
  - The RBF kernel is a function of the distance between two data points.
- $K(X_i,X_j)$:
  - This represents the kernel function applied to two data points $X_i, X_j$
- $||x_i-x_j||^2_2$
  - This is the squared Euclidean distance between two data points $x_i, x_j$
- $\sigma^2$
  - This is the variance parameter of the RBF kernel.
  - It determines the spread or scale of the kernel.
  - A small value makes the kernel more local (sensitive to small distances), while a large value makes it more global (less sensitive to distances).
- $N$: This is the total number of data points.
