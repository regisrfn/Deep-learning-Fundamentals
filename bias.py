import numpy as np
import costFunc

def adjust_bias(x,y,initial_weight=0.0,initial_bias = 0.0, learning_rate = 0.01):
    
    new_bias = initial_bias - learning_rate*costFunc.derivative_of_bias(x,y,initial_weight,initial_bias)

    return new_bias