import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.y = y
        self.x = x

        self.w1 = np.random.randn(self.x.shape[1], 10)
        self.b1 = np.zeros((1, self.w1.shape[1]))
        
        self.w2 = np.random.randn(10, self.y.shape[1])
        self.b2 = np.zeros((1, self.w2.shape[1]))

    def feedforward(self):
        self.a1 = sigmoid(np.dot(self.x, self.w1) + self.b1)
        self.a2 = sigmoid(np.dot(self.a1, self.w2) + self.b2)


    def backprop(self,learning_rate=0.1):
        N = len(self.x)
        
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        # DE/dw2 = DE/Da2 * Da2/Dz2 * Dz2/Dw2
        # DE/dw1 = DE/Da2 * Da2/Dz2 * Dz2/Da2 *Da1/Dz1 * Dz1/Dw1
        
        # DE/Dz2 = (a2 - y)
        DE_dz2 = (self.a2 - self.y)
        DE_dz1 = np.dot(DE_dz2,self.w2.T) * sigmoid_derivative(self.a1)

        # DE/Da2 * Da2/Dz2 * Dz2/Dw2
        d_weights2 = np.dot(self.a2.T,DE_dz2)
        d_bias2 = np.sum(DE_dz2,axis=0)
        d_weights1 = np.dot(self.x.T,DE_dz1)
        d_bias1 = np.sum(DE_dz1,axis=0)

        d_weights2 /= N
        d_weights2 *= learning_rate
        d_bias2 /= N
        d_bias2 *= learning_rate
        d_weights1 /= N
        d_weights1 *= learning_rate
        d_bias1 /= N
        d_bias1 *= learning_rate


        self.w2 -= d_weights2
        self.b2 -= d_bias2
        self.w1 -= d_weights1
        self.b1 -= d_bias1

    
    def predict(self,x_test):
        a1 = sigmoid(np.dot(x_test, self.w1) + self.b1)
        a2 = sigmoid(np.dot(a1, self.w2) + self.b2)
        return a2
