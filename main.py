import matplotlib.pyplot as plt
import numpy as np
import gradient_descent

# Fixing random state for reproducibility
# np.random.seed(19680801)

w = 2
b = 10
x = np.arange(0.0, 100, 1)
noise = np.random.normal(1, 10, x.shape)
y0 = w*x + b
y = y0 + noise

epochs = 1000
new_weight = 5.0
new_bias = 1.0

from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize([x])
normalized_X = normalized_X[0]

for epoch in range(epochs):
    new_weight, new_bias = gradient_descent.adjust_weights(normalized_X,y,new_weight,new_bias)    

print(new_weight,new_bias)
print(gradient_descent.get_cost(normalized_X,y,new_weight,new_bias))
print(gradient_descent.get_cost(x,y,w,b))

predicted_Y = new_weight*normalized_X + new_bias

plt.scatter(x, y)
plt.plot(x, y0)
plt.plot(x, predicted_Y)
plt.show()