# Importing the libraries
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logistic_regression.logistic_regression as classification

# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('./Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
Y = dataset.iloc[:, 4].values
Y = np.array(Y, ndmin=2).T

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


epochs = 1000

bias = np.ones(shape=(len(x_train), 1))
x_train = np.append(bias, x_train, axis=1)
bias = np.ones(shape=(len(x_test), 1))
x_test = np.append(bias, x_test, axis=1)
weights = np.zeros((x_train.shape[1], y_train.shape[1]))

for epoch in range(epochs):
    weights = classification.update_weights(
        x_train, y_train, weights, learning_rate=0.001)


print(weights.shape)

predicted_y = classification.predict(x_test, w=weights)
predicted_y = np.round(predicted_y)

cm = confusion_matrix(y_test, predicted_y)
print(cm)
print(f'accuracy:{classification.accuracy(predicted_y,y_test)}')

plt.scatter(x_test[:, 1], y_test)
plt.scatter(x_test[:, 1], predicted_y)
plt.show()
