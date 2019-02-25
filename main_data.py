# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradient_descent

# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('./Advertising.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, 4].values
Y = np.array(Y, ndmin=2).T

epochs = 1000
new_weight = np.array([
    [0],
    [0],
    [0]
])
new_bias = 1.0

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(X)



for epoch in range(1000):
    new_weight, new_bias = gradient_descent.adjust_weights(x_scaled,Y,new_weight, new_bias)


print(new_weight,new_bias)
print(gradient_descent.get_cost(x_scaled,Y,new_weight,new_bias))

predicted_Y = np.dot(x_scaled,new_weight) + new_bias
plt.scatter(X[:, 1], Y)
# plt.plot(x1, y0)
plt.plot(X[:, 1], predicted_Y)
plt.show()