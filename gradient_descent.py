import operator
import numpy as np
import sklearn.metrics
from matplotlib import pyplot
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# function: calc_gradient
# calculates the gradient for a specific weightVector
# returns: the mean of the gradients as a vector
def calc_gradient(weightVector, y_tild, X) :
    size = y_tild.shape[0]

    sum = 0
    for index in range(size) :
        sum += (-y_tild[index] * X[index]) / (1 + np.exp(y_tild[index] * weightVector.T * X[index]))
    mean = sum / size

    return mean

# function: gradient_descent
# calculates the gradient descent for a given X matrix with corresponding y vector
def gradient_descent( X, y, stepSize, maxIterations) :

    # declare weightVector which is initialized to the zero vector
    #   one element for each feature
    dimension = X.shape
    features = dimension[1]
    weightVector = np.zeros(features)

    # declare weightMatrix of real number
    #   number of rows = features, number of cols = maxIterations
    num_of_entries = features * maxIterations
    weightMatrix = np.array(np.zeros(num_of_entries).reshape(features, maxIterations))

    size = y.shape[0]
    y_tild = np.empty(size)
    for index in range(size):
        if (y[index] == 0): y_tild[index] = -1
        else : y_tild[index] = 1

    for index in range(maxIterations) :
        # first compute the gradient given the current weightVector
        #   make sure that the gradient is of the mean logistic loss over all training data
        #print(weightVector)
        gradient = calc_gradient(weightVector, y_tild, X)

        mean_grad_log_loss = gradient / X.shape[1]
        # then update weightVector by taking a step in the negative gradient direction
        weightVector = weightVector - stepSize * gradient

        weightMatrix[:, index] = weightVector[:]
        # then store the resulting weightVector in the corresponding column of weightMatrix
        #for row in range(features) :
        #    weightMatrix[row][index] = weightVector[row]

    return weightMatrix

# function sigmoid
def sigmoid(x) :
    x = 1 / (1 + np.exp(-x))
    return x
