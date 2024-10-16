import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression():
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        return (1/(1 + np.exp(-z)))

    def predict(self, X):
        # equation is y = sigmoid(mx+c)
        z = X*self.m + self.c
        return self.sigmoid(z)

    def train(self, X, y):
        self.m, self.c = 0, 0
        n = len(X)
        for i in range(self.epochs):
            y_pred = self.predict(X)
            log_loss = -1 * np.mean(y*np.log10(y_pred) +
                                    (1-y)*np.log10(1-y_pred))
            print("log_loss = ", log_loss)
            
            for j in range(n):
                # find partial derivatives of each coefficient
                dm =  (X[j] * (y[j]-self.sigmoid(np.dot([self.m],X[j])+self.c)))[0]
                dc = y[j]-self.sigmoid(np.dot(self.m,X[j])+self.c)
                # update m and c in direction of negative of derivative
                self.m += self.learning_rate * dm
                self.c += self.learning_rate * dc


train_X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
train_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(learning_rate=.001, epochs=1000)
model.train(train_X, train_y)
print("prediction = ", model.predict(train_X))

# plotting the model
plt.scatter(train_X, train_y)
plt.plot(train_X, model.predict(train_X))
plt.xlabel('train_x')
plt.ylabel('train_y')
plt.show()
