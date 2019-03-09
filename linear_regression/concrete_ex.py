# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradient_descent

# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('./concrete_data.csv')
X = dataset.iloc[:, 0:8].values
Y = dataset.iloc[:, 8].values
Y = np.array(Y, ndmin=2).T

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


epochs = 1000
bias = np.ones(shape=(len(x_train),1))
x_train = np.append(bias, x_train, axis=1)
bias = np.ones(shape=(len(x_test),1))
x_test = np.append(bias, x_test, axis=1)
weights = np.zeros((x_train.shape[1], 1))

for epoch in range(1000):
    weights = gradient_descent.adjust_weights(x_train,y_train,weights)


# print(new_weight,new_bias)
print(f'loss:{gradient_descent.get_cost(x_test,y_test,weights)}')

predicted_y = np.dot(x_test,weights)
plt.scatter(x_test[:, 1], y_test)
plt.scatter(x_test[:, 1], predicted_y)
plt.show()