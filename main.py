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

epochs = 1000
new_weight = np.array([
    [0],
    [0]
])
new_bias = 1.0

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


for epoch in range(epochs):
    new_weight, new_bias = gradient_descent.adjust_weights(x_scaled,y,new_weight, new_bias)    

print(gradient_descent.get_cost(x_scaled,y,new_weight,new_bias))
print(gradient_descent.get_cost(x,y,w,b))

predicted_Y = np.dot(x_scaled,new_weight) + new_bias

plt.scatter(x1, y)
plt.plot(x1, y0)
plt.plot(x1, predicted_Y)
plt.show()