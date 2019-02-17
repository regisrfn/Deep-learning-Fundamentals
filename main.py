import matplotlib.pyplot as plt
import numpy as np
import gradient_descent

# Fixing random state for reproducibility
# np.random.seed(19680801)

w = 2
b = 10
x = np.arange(0.0, 20, 1.0)
noise = np.random.normal(0, 1, x.shape)
y0 = w*x + b
y = y0 + noise

epochs = 1000
new_weight = 5
new_bias = 1

for i in range(1000):
    new_weight, new_bias = gradient_descent.adjust_weights(x,y,new_weight,new_bias)    

print(new_weight, new_bias)
print(gradient_descent.get_cost(x,y,new_weight,new_bias))
print(gradient_descent.get_cost(x,y,w,b))

predicted_Y = new_weight*x + new_bias

plt.scatter(x, y)
plt.plot(x, y0)
plt.plot(x, predicted_Y)
plt.show()