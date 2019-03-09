import matplotlib.pyplot as plt
import numpy as np

def get_cost(x, y, w):
    
    cost = 0.0
    m = x.size

    error = y-np.dot(x,w)

    cost = np.square(error)

    cost = cost.sum(axis=0)

    return cost/(2*m)

def adjust_weights(x,y, w, b=0, learning_rate = 0.01):
    N = len(x)
    
    # derivative
    error = y- np.dot(x,w)
    
    gradient = np.dot(-x.T, error)

    gradient /= N

    gradient *= learning_rate

    weights = w - gradient

    return weights


def sigmoid(x):
  return 1 / (1 + np.exp(-x))