import numpy as np
import sklearn.metrics
from sklearn.preprocessing import scale

# function: calc_gradient
# calculates the gradient for a specific weightVector
# returns: the mean of the gradients as a vector
def calc_gradient(weightVector, y, X) :
    sum = 0
    size = y.shape[0]
    for index in range(size) :
        y_tild = 0
        if(y[index] == 0) : y_tild = -1
        if(y[index] == 1) : y_tild = 1
        #print(X[index])
        #print("top" + str(-y_tild * X[index]))
        #print((1 + np.exp(y_tild * weightVector * X[index])))
        gradient = ((-y_tild * X[index]) / (1 + np.exp(y_tild * weightVector * X[index])))
        #print("grad" + str(gradient))
        sum += gradient
    return sum / size

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


    for index in range(maxIterations) :
        # first compute the gradient given the current weightVector
        #   make sure that the gradient is of the mean logistic loss over all training data
        gradient = calc_gradient(weightVector, y, X)

        # then update weightVector by taking a step in the negative gradient direction
        weightVector = weightVector - stepSize * gradient

        # then store the resulting weightVector in the corresponding column of weightMatrix
        for row in range(features) :
            weightMatrix[row][index] = weightVector[row]

    return weightMatrix

# read data from csv
all_data = np.genfromtxt('spam.data', delimiter=" ")
# get size of data
data_size = all_data.shape[1] - 1

# set inputs to everything but last col, and scale
inputs = np.delete(all_data, data_size, axis=1)
inputs_scaled = scale(inputs)
# set outputs to last col of data
outputs = all_data[:,data_size]

#print(all_data)
#print(inputs)
#print(inputs_scaled)
#print(outputs)

weightMatrix = gradient_descent(inputs_scaled, outputs, 0.1, 100)

print(inputs_scaled)

predicted = np.dot(inputs, weightMatrix)
print(predicted)

print(np.mean( outputs != predicted[:,0]))