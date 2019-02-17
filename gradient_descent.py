import matplotlib.pyplot as plt
import numpy as np

def get_cost(x, y, w = 0.5, bias = 0.0):
    
    cost = 0.0

    m = x.size
    for i in range (m):
        cost += np.square(y[i] - (w*x[i]+bias))

    return cost/(2*m)

def adjust_weights(x,y, w = 0.0, b = 0.0, learning_rate = 0.01):
    weight = 0.0
    bias = 0.0

    m = x.size
    for i in range (m):
        weight += -x[i]*(y[i] - (w*x[i]+b))
        bias += -(y[i] - (w*x[i]+b))

    derivative_of_weigth = weight/m
    derivative_of_bias = bias/m

    new_weight = w - learning_rate * derivative_of_weigth
    new_bias = b - learning_rate * derivative_of_bias


    return (new_weight, new_bias)


