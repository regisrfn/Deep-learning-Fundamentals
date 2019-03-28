import numpy as np
OUTPUT_LAYERS = 0
WEIGHTS = []
dW = []


def predict(x, w):
    z = np.dot(x, w)
    return sigmoid(z)


def cost_function(x, y, weights):

    N = len(y)

    predictions = predict(x, weights)

    # Take the error
    cost = -y*np.log(predictions) - (1-y)*np.log(1-predictions)

    # Take the average cost
    cost = cost.sum() / N

    return cost


def update_weights(x, y, learning_rate=0.01):
    N = len(x)
    w2 = WEIGHTS[1]
    w1 = WEIGHTS[0]

    # cost = E = -(ylog(a2) - (1-y)log(a2))

    z1 = np.dot(x, w1)
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)

    dE_dz2 = (a2-y)

    dE_dw2 = np.dot(a1.T, dE_dz2)
    dE_dw1 = np.dot(x.T,  np.dot(dE_dz2, w2.T) * (a1*(1-a1)))

    dE_dw2 /= N
    dE_dw2 *= learning_rate

    dE_dw1 /= N
    dE_dw1 *= learning_rate

    WEIGHTS[1] -= dE_dw2
    WEIGHTS[0] -= dE_dw1


def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


def feed_forward(x):

    for weight in WEIGHTS:
        yHat = predict(x, weight)
        x = yHat

    return yHat


def add_layer(size=0, input=0):

    if input > 0:
        last_layer_size = input
        w = np.zeros((last_layer_size, size))
        WEIGHTS.append(w)
    else:
        last_layer_size = WEIGHTS[0].shape[1]
        w = np.zeros((last_layer_size, size))
        WEIGHTS.append(w)

    return WEIGHTS


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
