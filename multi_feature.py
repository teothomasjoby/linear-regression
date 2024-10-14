import numpy as np

class LinearRegression:
    def __init__(self):
        self.W = [0.0] * 7

    def predict(self, X):
        predictions = []
        for row in X:
            prediction = self.W[0] 
            for i in range(len(row)):
                prediction += self.W[i + 1] * row[i]
            predictions.append(prediction)
        return predictions
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        m = len(y)
        self.W = np.zeros(7)

        for epoch in range(epochs):
            mse = 1/m * np.sum((self.predict(X)-y)**2)
            print(mse)
            weights = []
            for i in range(m):
                y_pred = self.predict([X[i]])[0]
                weights[0] = 1/m * np.sum((self.predict(X)-y)**2)
                for j in range()
                error = y_pred - y[i]
            #     gradients[0] += error  # Gradient for bias term
            #     for j in range(6):
            #         gradients[j + 1] += error * X[i][j]  # Gradient for feature j
            
            # # Update weights
            # for k in range(7):
            #     self.weights[k] -= (learning_rate / m) * gradients[k]

X = np.array([
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 9]])

y = np.array([10, 12, 14, 16])

model = LinearRegression()
model.fit(X, y, epochs=1, learning_rate=0.001)

predictions = model.predict(X)
print("Predictions:", predictions)
