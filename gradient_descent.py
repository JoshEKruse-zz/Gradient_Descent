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


def driver(X_train, y_train, X_test, y_test, X_val, y_val):
    # print sizes of each set
    print("{: >11} {: >4} {: >4}".format("", "y", ""))
    print("{: >11} {: >4} {: >4}".format("set", 0, 1))
    print("{: >11} {: >4} {: >4}".format("train", np.sum(y_train == 0), np.sum(y_train == 1)))
    print("{: >11} {: >4} {: >4}".format("test", np.sum(y_test == 0), np.sum(y_test == 1)))
    print("{: >11} {: >4} {: >4}".format("validation", np.sum(y_val == 0), np.sum(y_val == 1)))

    # get weightMatrix
    maxIterations = 500
    weightMatrix = gradient_descent(X_train, y_train, 0.05, maxIterations)

    # vectorize sigmoid function to pass on prediction matrix
    sig_v = np.vectorize(sigmoid)

    # calculate predicted matrix, round each answer
    train_pred = np.dot(X_train, weightMatrix)
    val_pred = np.dot(X_val, weightMatrix)
    test_pred = np.dot(X_test, weightMatrix)

    # pass sigmoid function onto prediction and round
    train_pred = np.around(sig_v(train_pred))
    val_pred = np.around(sig_v(val_pred))
    test_pred = np.around(sig_v(test_pred))

    # calculate percent error
    train_result = []
    val_result = []
    for index in range(maxIterations):
        train_result.append(np.mean(y_train != train_pred[:, index]))
    for index in range(maxIterations):
        val_result.append(np.mean(y_val != val_pred[:, index]))
    train_min_index, train_min_value = min(enumerate(train_result), key=operator.itemgetter(1))
    val_min_index, val_min_value = min(enumerate(val_result), key=operator.itemgetter(1))

    print("Number of iterations to minimize the validation error: " + str(val_min_index))

    # graph percent error
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(train_result, "g-", label='Train')
    ax.annotate('train min', xy=(train_min_index, train_min_value), xytext=(train_min_index, train_min_value),
                arrowprops=dict(facecolor='green', shrink=0.05), )
    line2, = ax.plot(val_result, "r-", label='Validation')
    ax.annotate('validation min', xy=(val_min_index, val_min_value), xytext=(val_min_index, val_min_value),
                arrowprops=dict(facecolor='red', shrink=0.05), )
    ax.set_ylabel("Percent")
    ax.set_xlabel("Iterations")
    ax.set_title("Percent Error")
    # ax.set_ylim(0,.8)
    ax.legend()
    pyplot.show()

    # calculate logistic loss
    train_loss_results = []
    for index in range(maxIterations):
        train_loss_results.append(sklearn.metrics.log_loss(y_train, train_pred[:, index]))
    val_loss_results = []
    for index in range(maxIterations):
        val_loss_results.append(sklearn.metrics.log_loss(y_val, val_pred[:, index]))
    train_min_index, train_min_value = min(enumerate(train_loss_results), key=operator.itemgetter(1))
    val_min_index, val_min_value = min(enumerate(val_loss_results), key=operator.itemgetter(1))

    print(train_loss_results)
    print(val_loss_results)

    # plot logistic loss
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(train_loss_results, "g-", label='Train')
    ax.annotate('train min', xy=(train_min_index, train_min_value), xytext=(train_min_index, train_min_value),
                arrowprops=dict(facecolor='green', shrink=0.05), )
    line2, = ax.plot(val_loss_results, "r-", label='Validation')
    ax.annotate('validation min', xy=(val_min_index, val_min_value), xytext=(val_min_index, val_min_value),
                arrowprops=dict(facecolor='red', shrink=0.05), )

    ax.set_ylabel("Loss")
    ax.set_xlabel("Iterations")
    ax.set_title("Logistic Loss")
    # ax.set_ylim(0,.8)
    ax.legend()
    pyplot.show()

    # use minimum validation to grab best prediction vector
    best_tr_pred = train_pred[:, val_min_index]
    best_va_pred = val_pred[:, val_min_index]
    best_te_pred = test_pred[:, val_min_index]

    # calculate logistic regression train error
    train_error = int(np.mean(y_train != best_tr_pred) * 100)
    val_error = int(np.mean(y_val != best_va_pred) * 100)
    test_error = int(np.mean(y_test != best_te_pred) * 100)

    # create zero vector for y baseline
    y_train_baseline = np.zeros(best_tr_pred.shape[0])
    y_val_baseline = np.zeros(best_va_pred.shape[0])
    y_test_baseline = np.zeros(best_te_pred.shape[0])

    # calculate logistic regression train error
    baseline_train_error = int(np.mean(y_train != y_train_baseline) * 100)
    baseline_val_error = int(np.mean(y_val != y_val_baseline) * 100)
    baseline_test_error = int(np.mean(y_test != y_test_baseline) * 100)

    # create log reg table of errors, baseline errors
    print("{: >11} {: >9} {: >9}".format("", "log reg", "baseline"))
    print("{: >11} {: >9} {: >9}".format("train", str(train_error) + "%", str(baseline_train_error) + "%"))
    print("{: >11} {: >9} {: >9}".format("validation", str(val_error) + "%", str(baseline_val_error) + "%"))
    print("{: >11} {: >9} {: >9}".format("test", str(test_error) + "%", str(baseline_test_error) + "%"))

    return weightMatrix, val_min_index, y_test_baseline

# read data from csv
all_data = np.genfromtxt('spam.data', delimiter=" ")
# get size of data
size = all_data.shape[1] - 1
# set inputs to everything but last col, and scale
X = scale(np.delete(all_data, size, axis=1))
# set outputs to last col of data
y = all_data[:, size]

pyplot.figure()

log_roc = []

for index in range(3):
    # Create train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=index)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=index)

    weightMatrix, val_min_index, y_test_baseline = driver(X_train, y_train, X_test, y_test, X_val, y_val)

    # vectorize sigmoid function to pass on prediction matrix
    sig_v = np.vectorize(sigmoid)

    # calculate roc curves for logistic regression and baseline
    fpr_log, tpr_log, _ = roc_curve(y_test, sig_v(np.dot(X_test, weightMatrix))[:, val_min_index])
    log_roc.append((fpr_log, tpr_log))

fpr_base, tpr_base, _ = roc_curve(y_test, y_test_baseline, pos_label=1)

pyplot.plot(log_roc[0][0], log_roc[0][1], "b-", label="Logistic Regression Set 1") # plot logistic regression
pyplot.plot(log_roc[1][0], log_roc[1][1], "b-", label="Logistic Regression Set 2") # plot logistic regression
pyplot.plot(log_roc[2][0], log_roc[2][1], "b-", label="Logistic Regression Set 3") # plot logistic regression

pyplot.plot(fpr_base, tpr_base, "m-", label="Baseline") # plot baseline
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.title('Receiver Operating Characteristic')
pyplot.legend(loc="lower right")
pyplot.show()