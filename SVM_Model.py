'''
Do Not compile the file
It just used for verify
the structure of project
'''

# Importing useful libraries and modules

import os
import pandas as pd
import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers
from cvxopt import matrix
from cvxopt import solvers
from numpy import array
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# %matplotlib inline
np.random.seed(40)


# Reading the files and converting to necessary labels -1 and 1

X_train, Y_test = pd.read_csv('../Data/train.csv', header=None), pd.read_csv('../Data/test.csv', header=None)
X_data_train, X_data_val = X_train.iloc[:7225, 1:].to_numpy(), X_train.iloc[7225:, 1:].to_numpy()

y_label_train, y_label_val = X_train.iloc[:7225, 0].to_numpy(), X_train.iloc[7225:, 0].to_numpy()

Y, y = Y_test.iloc[:, 1:].to_numpy(), Y_test.iloc[:, 0].to_numpy()

y_label_train, y_label_val, y = np.where(y_label_train == 0, -1, y_label_train), 
np.where(y_label_val == 0, -1, y_label_val), np.where(y == 0, -1, y)

# SVM Primal Form - training function


def svm_train_primal(data_train, label_train, regularisation_para_C):
    a, b = data_train.shape
    P_w,  P_s, P_b = np.asarray(np.diag(np.ones(b))), np.zeros([
        b, a]),  np.zeros([1, b])

    P_1, P_2, P_3, P_4 = np.zeros([a, a]), np.zeros(
        [a, 1]), np.zeros([1, a]), np.zeros([1, 1])
    P = cvxopt.matrix(np.concatenate((np.concatenate((P_w, P_s, P_b.T), axis=1), np.concatenate(
        (P_s.T, P_1, P_2), axis=1), np.concatenate((P_b, P_3, P_4), axis=1)), axis=0))

    q = cvxopt.matrix((np.concatenate(
        (P_b, np.matrix(np.ones([a]) * regularisation_para_C/a), P_4), axis=1).T))

    g_1, g_2, g_3 = np.dot(np.diag(label_train), data_train) * (-1), np.asarray(
        np.diag(np.ones([a]) * (-1))),  np.matrix(label_train) * (-1)
    g_3 = g_3.T

    G = np.vstack((np.concatenate((g_1, g_2, g_3), axis=1), np.concatenate(
        (np.zeros([a, b + 1]), np.asarray(np.diag(np.ones([a]) * (-1)))), axis=1)))
    G = cvxopt.matrix(G)

    h = cvxopt.matrix(np.hstack((np.ones(a) * (-1), np.zeros(a))))
    svm_model = np.array(cvxopt.solvers.coneqp(P, q, G, h)['x']).flatten()
    return svm_model

# SVM Primal Form - testing function


def svm_predict_primal(data_test, label_test, svm_model):
    a, b = X_data_train.shape
    accuracy = accuracy_score(label_test, np.sign(
        np.dot(data_test, svm_model[:b]) + svm_model[-1]))
    return accuracy


regularisation_para_C = [1] + list(range(0, 101, 5))

for k in regularisation_para_C:
    svm_model = svm_train_primal(X_data_train, y_label_train, k)
    test_accuracy_primal = svm_predict_primal(
        X_data_val, y_label_val, svm_model)
    print('The training accuracy for C = ', k, ' is: ', test_accuracy_primal)

svm_model = svm_train_primal(X_data_train, y_label_train, 100)
train_accuracy, val_accuracy, test_accuracy = svm_predict_primal(X_data_train, y_label_train, svm_model), svm_predict_primal(
    X_data_val, y_label_val, svm_model), svm_predict_primal(Y, y, svm_model)

print("The train accuracy for our training set: ", train_accuracy)
print("The val accuracy for our validation set: ", val_accuracy)
print("The test accuracy for our testing set: ", test_accuracy)

w_primal, b_primal = svm_model[:X_data_train.shape[1]], svm_model[-1]
primal_W_series, primal_b_series = pd.Series(w_primal), pd.Series(b_primal)
primal_series = pd.concat([primal_W_series, primal_b_series], axis=0)


# SVM Dual Form - training function

def svm_train_dual(data_train,  label_train,  regularisation_para_C):
    a, b = X_data_train.shape
    label_train = label_train.reshape(-1, 1) * 1.

    label_train_diagonal = np.asarray(
        np.diag(np.ones(label_train.shape[0]))*label_train)
    mat = np.dot(label_train_diagonal, data_train)
    mat.shape

    P, q = matrix(np.dot(mat, mat.T) * 1.), matrix(-np.ones((a, 1)))

    G = matrix(np.vstack((np.eye(a) * (-1), np.eye(a))))
    h = matrix(np.hstack((np.zeros(a), np.ones(a) * regularisation_para_C/a)))

    A, b = matrix(label_train.reshape(1, -1)), matrix(np.zeros(1))
    res = solvers.qp(P, q, G, h, A, b)
    w_1 = (label_train * np.array(res['x'])).T @ data_train
    w = w_1.reshape(-1, 1)

    svm_model_d = np.vstack((w_1.reshape(-1, 1), label_train[(np.array(res['x']) > 1e-4).flatten(
    )] - np.dot(data_train[(np.array(res['x']) > 1e-4).flatten()], w).mean()))
    return svm_model_d

# SVM Dual Form - testing function


def svm_predict_dual(data_test, label_test, svm_model_d):
    a, b = X_data_train.shape
    accuracy = accuracy_score(label_test, np.sign(np.dot(
        svm_model_d[:b].T, data_test.T) + svm_model_d[-1]).reshape(label_test.shape))
    return accuracy


regularisation_para_C = [1] + list(range(0, 101, 5))
for k in regularisation_para_C:
    svm_model_dual = svm_train_dual(X_data_train, y_label_train, k)
    test_accuracy_dual = svm_predict_dual(
        X_data_val, y_label_val, svm_model_dual)
    print('The training accuracy for C = ', k, ' is: ', test_accuracy_dual)

svm_model_d = svm_train_dual(X_data_train, y_label_train, 1)
train_accuracy, val_accuracy, test_accuracy = svm_predict_dual(X_data_train, y_label_train, svm_model_d), svm_predict_dual(
    X_data_val, y_label_val, svm_model_d), svm_predict_dual(Y, y, svm_model_d)

print("The train accuracy for our train set: ", train_accuracy)
print("The val accuracy for our validation set: ", val_accuracy)
print("The test accuracy for our testing set: ", test_accuracy)

W_dual, b_dual = svm_model_d[:X_data_train.shape[1]], svm_model_d[-1]
dual_W_series, dual_b_series = pd.Series(W_dual.flatten()), pd.Series(b_dual)
dual_series = pd.concat((dual_W_series, dual_b_series), axis=0)


# SVM using Existing Libraries - Scikit-learn

regularisation_para_C = list(range(1, 101, 5))
regularisation_para_C = [i / X_data_train.shape[0]
                         for i in regularisation_para_C]


for j in regularisation_para_C:
    SVM_classifier = SVC(C=j, kernel='linear')
    SVM_classifier.fit(X_data_train, y_label_train)
    predicted_labels = SVM_classifier.predict(X_data_val)
    accuracy = accuracy_score(y_label_val, predicted_labels)
    print('The Training accuracy for C = ', j *
          X_data_train.shape[0], ' is: ', accuracy)

SVM_classifier = SVC(C=100 / X_data_train.shape[0], kernel='linear')
SVM_classifier.fit(X_data_train, y_label_train)
predicted_labels_train = SVM_classifier.predict(X_data_train)
train_accuracy = accuracy_score(y_label_train, predicted_labels_train)
predicted_labels_test = SVM_classifier.predict(Y)
test_accuracy = accuracy_score(y, predicted_labels_test)
print('The overall training accuracy is : ', train_accuracy)
print('The overall testing accuracy is : ', test_accuracy)


SVC_series = pd.concat((pd.Series(SVM_classifier.coef_.flatten()), pd.Series(
    SVM_classifier.intercept_)), axis=0)

parameter_comparison = pd.concat([dual_series.rename(
    'SVM Dual Form'), primal_series.rename('SVM Primal Form')], axis=1)
parameter_difference = round(
    (parameter_comparison['SVM Primal Form'] - parameter_comparison['SVM Dual Form']), 4)
parameter_difference_percentage = (
    parameter_comparison['SVM Primal Form'] - parameter_comparison['SVM Dual Form'])/parameter_comparison['SVM Dual Form']*100
final_parameter_comparison = pd.concat(
    [parameter_comparison, parameter_difference, parameter_difference_percentage], axis=1)
weights_comparison = final_parameter_comparison.iloc[:200]

plt.figure(figsize=(14, 8), dpi=250, facecolor='gray', edgecolor='blue')
plt.scatter(range(0, 201),
            final_parameter_comparison['SVM Primal Form'], c='red', marker='d')
plt.scatter(range(0, 201),
            final_parameter_comparison['SVM Dual Form'], c='yellow', marker='+')
plt.title('Comparision of Weights and Bias for Primal and Dual Form')
