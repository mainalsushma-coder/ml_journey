import numpy as np

class LinearRegressionScratch:

    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations

    def fit(self, X, y):
        m = 0
        b = 0
        n = len(X)

        self.losses = []

        for _ in range(self.iterations):
            y_pred = m*X + b

            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            m = m - self.lr * dm
            b = b - self.lr * db

            loss = np.mean((y - y_pred)**2)
            self.losses.append(loss)

        self.m = m
        self.b = b

    def predict(self, X):
        return self.m * X + self.b