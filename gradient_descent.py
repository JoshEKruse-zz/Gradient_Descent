import operator

import numpy as np
import sklearn.metrics
from matplotlib import pyplot
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# function: calc_gradient
# calculates the gradient for a specific weightVector
# returns: the mean of the gradients as a vector
def calc_gradient(weightVector, y, X) :
    size = y.shape[0]
    y_tild = np.empty(size)
    for index in range(size) :
        if(y[index] == 0) : y_tild[index] = -1
        if(y[index] == 1) : y_tild[index] = 1

    sum = 0
    for index in range(size) :
        sum += (-y_tild[index] * X[index]) / (1 + np.exp(y_tild[index] * weightVector * X[index]))
    mean = sum / size

    #m = X.shape[0]
    #gradient =  (1 / m) * np.dot(X.T, (1 / (1 + np.exp(-(np.dot(X, weightVector))))) - y)

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


    for index in range(maxIterations) :
        # first compute the gradient given the current weightVector
        #   make sure that the gradient is of the mean logistic loss over all training data
        #print(weightVector)
        gradient = calc_gradient(weightVector, y, X)

        # then update weightVector by taking a step in the negative gradient direction
        weightVector = weightVector - stepSize * gradient

        # then store the resulting weightVector in the corresponding column of weightMatrix
        for row in range(features) :
            weightMatrix[row][index] = weightVector[row]

    return weightMatrix

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

# read data from csv
all_data = np.genfromtxt('spam.data', delimiter=" ")
# get size of data
size = all_data.shape[1] - 1
# set inputs to everything but last col, and scale
X = scale(np.delete(all_data, size, axis=1))
# set outputs to last col of data
y = all_data[:, size]

# Create train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# print sizes of each set
print("           " + "y    " + "    ")
print("set        " + str(0) + "    " + str(1))
print("train      " + str(np.sum(y_train == 0)) + " " + str(np.sum(y_train == 1)))
print("test       " + str(np.sum(y_test == 0)) + "  " + str(np.sum(y_test == 1)))
print("validation " + str(np.sum(y_val == 0)) + "  " + str(np.sum(y_val == 1)))

# get weightMatrix
maxIterations = 1000
weightMatrix = gradient_descent(X_train, y_train, 0.4, maxIterations)

# calculate predicted matrix, round each answer
train_pred = np.around(np.dot( X_train, weightMatrix))
val_pred = np.around(np.dot( X_val, weightMatrix))

# calculate percent error
train_result = []
val_result = []
for index in range(maxIterations) :
    train_result.append( np.mean( y_train == train_pred[:,index]))
for index in range(maxIterations):
    val_result.append(np.mean( y_val == val_pred[:, index]))
train_min_index, train_min_value = min(enumerate(train_result), key=operator.itemgetter(1))
val_min_index, val_min_value = min(enumerate(val_result), key=operator.itemgetter(1))

fig = pyplot.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(train_result, label='train')
ax.annotate('train min', xy=(train_min_index, train_min_value), xytext=(train_min_index, train_min_value),
            arrowprops=dict(facecolor='black', shrink=0.05),)
line2, = ax.plot(val_result, label='validation')
ax.annotate('validation min', xy=(val_min_index, val_min_value), xytext=(val_min_index, val_min_value+.01),
            arrowprops=dict(facecolor='black', shrink=0.05),)
ax.set_ylabel("Percent")
ax.set_xlabel("Iterations")
ax.set_title("Percent Error")
ax.set_ylim(0,.8)
ax.legend()
pyplot.show()

# calculate logistic loss
train_results = []
for index in range(maxIterations) :
    train_results.append( sklearn.metrics.log_loss( y_train, train_pred[:,index]) )
val_results = []
for index in range(maxIterations):
    val_results.append(sklearn.metrics.log_loss( y_val, val_pred[:, index]))
train_min = min(train_result)
val_min = min(val_result)
train_min_index, train_min_value = min(enumerate(train_result), key=operator.itemgetter(1))
val_min_index, val_min_value = min(enumerate(val_result), key=operator.itemgetter(1))

fig = pyplot.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(train_result, label='train')
ax.annotate('train min', xy=(train_min_index, train_min_value), xytext=(train_min_index, train_min_value),
            arrowprops=dict(facecolor='black', shrink=0.05),)
line2, = ax.plot(val_result, label='validation')
ax.annotate('validation min', xy=(val_min_index, val_min_value), xytext=(val_min_index, val_min_value+.01),
            arrowprops=dict(facecolor='black', shrink=0.05),)
ax.set_ylabel("Loss")
ax.set_xlabel("Iterations")
ax.set_title("Logistic Loss")
ax.set_ylim(0,.8)
ax.legend()
pyplot.show()