import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.model import LinearRegressionScratch
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

model = LinearRegressionScratch(lr=0.01, iterations=1000)
model.fit(x, y)

plt.plot(model.losses)
plt.title("Loss vs Iterations")
plt.show()