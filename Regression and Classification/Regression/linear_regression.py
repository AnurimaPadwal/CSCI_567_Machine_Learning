"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    N = np.shape(X)[0]
    y_pred = np.dot(X,w)
    err = np.abs(y_pred - y)
    err = np.sum(err)/N
    

    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  temp = np.linalg.inv(np.dot(X.T, X))
  w = np.dot(np.dot(temp, X.T), y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    mat = np.dot(X.T, X)
    D = np.shape(X)[1]
    
    while(1):
        w, v = np.linalg.eig(mat)
        if(np.any(np.abs(w) < 10 ** (-5))):
            mat = mat + 0.1 * np.identity(D)
        else:
            break
    mat = np.linalg.inv(mat)
    w = np.dot(np.dot(mat, X.T), y)
    
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################
    D = np.shape(X)[1]
    mat =  np.linalg.inv(np.add(np.dot(X.T, X), lambd * np.identity(D)))
    w = np.dot(np.dot(mat, X.T), y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    min_MAE = 9999
    lambds = [10 ** pow for pow in range(-19,20)]
    for param in lambds:
        w = regularized_linear_regression(Xtrain, ytrain, param)
        MAE = mean_absolute_error(w, Xval, yval)
        #print("param " + str(param))
        #print("MAE " + str(MAE))
        if MAE < min_MAE:
            bestlambda = param
            min_MAE = MAE
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    temp = np.copy(X)
    for i in range(2, power+1):
        pow_mat = np.power(temp,i)
        X = np.hstack((X, pow_mat))
    return X


