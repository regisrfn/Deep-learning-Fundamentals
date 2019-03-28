import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.y = y

        self.x = x
        bias = np.ones((self.x.shape[0], 1))
        self.x = np.append(bias, self.x, axis=1)

        self.w1 = np.zeros((self.x.shape[1], self.y.shape[1]))

    def feedforward(self):
        self.a1 = sigmoid(np.dot(self.x, self.w1))

    def backprop(self,learning_rate=0.01):
        N = len(self.x)
        
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        # DE/dw2 = DE/Da2 * Da2/Dz2 * Dz2/Dw2
        # DE/dw2 = DE/Da2 * Da2/Dz2 * Dz2/Da1 *Da1/Dz1 * Dz1/Dw1
        
        # DE/Dz2 = (a2 - y)
        DE_dz2 = (self.a1 - self.y)

        # DE/Da2 * Da2/Dz2 * Dz2/Dw2
        d_weights1 = np.dot(self.x.T, DE_dz2)

        d_weights1 /= N
        d_weights1 *= learning_rate

        self.w1 -= d_weights1
    
    def predict(self,x_test):
        bias = np.ones((x_test.shape[0], 1))
        x_test = np.append(bias, x_test, axis=1)
        y = sigmoid(np.dot(x_test, self.w1))
        return y
