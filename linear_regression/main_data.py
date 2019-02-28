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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


epochs = 1000
new_weight = np.zeros((x_train.shape[1], 1))
new_bias = 1.0
for epoch in range(1000):
    new_weight, new_bias = gradient_descent.adjust_weights(x_train,y_train,new_weight, new_bias)


print(new_weight,new_bias)
print(gradient_descent.get_cost(x_test,y_test,new_weight,new_bias))

predicted_y = np.dot(x_test,new_weight) + new_bias
# predicted_y = gradient_descent.sigmoid(predicted_y)
# predicted_y = np.round(predicted_y)

plt.scatter(x_test[:, 1], y_test)
plt.scatter(x_test[:, 1], predicted_y)
plt.show()