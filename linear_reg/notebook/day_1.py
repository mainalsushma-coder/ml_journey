import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])
m = 0.8
b = 1

y_pred = m*x + b


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

print("MSE:", mse(y, y_pred))
