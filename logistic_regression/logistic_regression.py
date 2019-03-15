import matplotlib.pyplot as plt
import numpy as np

def predict(x,w):
    z = np.dot(x, w)
    return sigmoid(z)

def cost_function(x, y, weights):

    N = len(y)

    predictions = predict(x, weights)

    #Take the error
    cost = -y*np.log(predictions) - (1-y)*np.log(1-predictions)

    #Take the average cost
    cost = cost.sum() / N

    return cost

def update_weights(x,y, w,learning_rate = 0.01):
    N = len(x)

    for category in range(y.shape[1]):
      label = y[:,category]
      label = np.array(label, ndmin=2).T
      # derivative
      predictions = predict(x,w)
      
      gradient = np.dot(x.T, predictions-label)
  
      gradient /= N
  
      gradient *= learning_rate
  
      weights = w - gradient

    return (weights)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))