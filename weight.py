import numpy as np
import costFunc

def adjust_W(x,y,initial_weight = 0.0, initial_bias = 0.0, learning_rate = 0.01):
    
    new_weight = initial_weight - learning_rate*costFunc.derivative_of_weigth(x,y,initial_weight,initial_bias)
    return new_weight