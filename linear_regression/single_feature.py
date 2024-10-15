import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        # equation is y = mx+c
        self.m, self.c = 0, 0
        n = len(X)
        for i in range(self.epochs):
            y_pred = X*self.m + self.c
            mse = 1/n * np.sum((y_pred - y) ** 2)
            print("mse = ", mse)
            # find partial derivatives of each coefficient
            dm = 2/n * np.sum((y_pred - y) * X)
            dc = 2/n * np.sum((y_pred - y))
            # update m and c in direction of negative of derivative
            self.m -= self.learning_rate * dm
            self.c -= self.learning_rate * dc
            print("m = ", self.m, "c = ", self.c)

    def predict(self, x):
        y = self.m*x + self.c
        return y

train_X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
train_y = np.array([2, 4, 6, 8, 10, 12, 14, 16])

model = LinearRegression(learning_rate=.001, epochs=100)
model.train(train_X, train_y)
print("prediction = ", model.predict(4))

# plotting the model
plt.scatter(train_X, train_y)
plt.plot(train_X, model.m * train_X + model.c)
plt.xlabel('train_x')
plt.ylabel('train_y')
plt.show()