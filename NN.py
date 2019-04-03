from sklearn.metrics import confusion_matrix
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import neural_network.NeuralNework as classification

# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plt.imshow(X_train[0])
# plt.show()

# flatten images into one-dimensional vector

# find size of one-dimensional vector
num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype(
    'float32')  # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype(
    'float32')  # flatten test images

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)

num_classes = y_train.shape[1]
num_features = num_pixels

nn = classification.NeuralNetwork(X_train,y_train)

epochs = 5000
for epoch in range(epochs):
    nn.feedforward()
    nn.backprop(learning_rate=0.5)
    print(f"epoch {epoch}/{epochs}")
    predicted_y = nn.predict(X_test)
    predicted_y = predicted_y.argmax(axis=1)
    print(f"accuracy: {np.mean(predicted_y == y_test)}")

