import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:

    def predict(self, X):
        predictions = []
        for row in X:
            prediction = self.W[0]
            for i in range(len(row)):
                prediction += self.W[i + 1] * row[i]
            predictions.append(prediction)
        return predictions

    def fit(self, X, y, epochs=1000, learning_rate=0.001):
        m, n = X.shape
        self.W = np.zeros(n+1)

        for epoch in range(epochs):
            error = self.predict(X)-y
            mse = 1/m * np.sum(error**2)
            print(mse)
            dw = np.zeros(n+1)
            dw[0] = 2/m * np.sum(error)  # partial derivative for bias

            # calculate partial derivative for other weights
            for j in range(n):
                dw[j + 1] = 2 / m * np.sum(error * X[:, j])
                # X[:, j] is the vector of all X values in the data

            # updating weights
            self.W[0] -= (learning_rate) * dw[0]
            for k in range(1, n+1):
                self.W[k] -= (learning_rate) * dw[k]

    def plot_predictions(self, predictions, y):
        plt.scatter(range(len(y)), y, color='blue', label='Actual Values')
        plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Values', marker='x')
        plt.xlabel('Data Points')
        plt.ylabel('Target Values')
        plt.title('Model Predictions vs. Actual Values')
        plt.legend()
        plt.show()

X = np.array([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 9]])

y = np.array([10, 12, 14, 16])

model = LinearRegression()
model.fit(X, y, epochs=1000, learning_rate=0.001)

predictions = model.predict(X)
print("Predictions:", predictions)

model.plot_predictions(predictions, y)

