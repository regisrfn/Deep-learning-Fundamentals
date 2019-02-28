import matplotlib.pyplot as plt
import numpy as np

def get_cost(x, y, w = 0.5, bias = 0.0):
    
    cost = 0.0
    m = x.size

    error = (y-bias) - np.dot(x,w)
    cost = np.square(error)

    cost = cost.sum(axis=0)

    return cost/(2*m)

def adjust_weights(x,y, w, b=0, learning_rate = 0.01):
    N = len(x)
    
    # derivative
    error = (y-b) - np.dot(x,w)
    
    gradient = np.dot(-x.T, error)
    gradient_bias = -np.sum(error)

    gradient /= N
    gradient_bias /= N

    gradient *= learning_rate
    gradient_bias *= learning_rate

    weights = w - gradient
    bias = b - gradient_bias

    return (weights, bias)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))