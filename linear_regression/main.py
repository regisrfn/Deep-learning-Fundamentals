import matplotlib.pyplot as plt
import numpy as np
import gradient_descent

# Fixing random state for reproducibility
# np.random.seed(19680801)

w = np.array([
    [2.],
    [2.]
])

b = 50

x1 = np.arange(0.0, 10, 1)
x2 = np.arange(0.0, 10, 1)
x =  np.array([x1,x2]).T


y0 = np.dot(x,w) + b
noise = np.random.normal(1, 1, y0.shape)
y = y0 + noise

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x)

epochs = 1000
bias = np.ones(shape=(len(x_train),1))
x_train = np.append(bias, x_train, axis=1)
weights = np.zeros((x_train.shape[1], 1))

for epoch in range(epochs):
    weights = gradient_descent.adjust_weights(x_train,y,weights)    

print(gradient_descent.get_cost(x_train,y,weights))
print(gradient_descent.get_cost(x,y-b,w))

predicted_Y = np.dot(x_train,weights)

plt.scatter(x1, y)
plt.plot(x1, y0)
plt.plot(x1, predicted_Y)
plt.show()