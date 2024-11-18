import numpy as np

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01, epochs=1000):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.parameters = self.initialize_parameters()
    
    def initialize_parameters(self):
        np.random.seed(42)
        parameters = {
            'W1': np.random.randn(self.n_hidden, self.n_input) * 0.01,
            'b1': np.zeros((self.n_hidden, 1)),
            'W2': np.random.randn(self.n_output, self.n_hidden) * 0.01,
            'b2': np.zeros((self.n_output, 1))
        }
        return parameters
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def forward_propagation(self, X):
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']

        z1 = np.dot(W1, X) + b1
        A1 = self.relu(z1)
        z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(z2)   

        cache = {'Z1': z1, 'A1': A1, 'Z2': z2, 'A2': A2}
        return A2, cache
    
    def compute_loss(self, y, A2):
        m = y.shape[1]
        loss = -np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m
        return np.squeeze(loss)
    
    def backward_propagation(self, X, y, cache):
        m = X.shape[1]
        W2 = self.parameters['W2']
        
        A1, A2 = cache['A1'], cache['A2']
        Z1 = cache['Z1']
        
        dZ2 = A2 - y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        
        dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        
        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads
    
    def update_parameters(self, grads):
        self.parameters['W1'] -= self.learning_rate * grads['dW1']
        self.parameters['b1'] -= self.learning_rate * grads['db1']
        self.parameters['W2'] -= self.learning_rate * grads['dW2']
        self.parameters['b2'] -= self.learning_rate * grads['db2']

    def train(self, X, y):
        for i in range(self.epochs):
            A2, cache = self.forward_propagation(X)
            loss = self.compute_loss(y, A2)
            grads = self.backward_propagation(X, y, cache)
            self.update_parameters(grads)
            
            if i % 100 == 0:
                print(f'Epoch {i}, Loss: {loss}')
    
    def predict(self, X):
        A2, _ = self.forward_propagation(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions
    

np.random.seed(42)
X = np.random.rand(2, 500)
y = (X[0] + X[1] > 1).astype(int).reshape(1, 500)

# Instantiate and train the neural network
nn = NeuralNetwork(n_input=2, n_hidden=3, n_output=1, learning_rate=0.01, epochs=10000)
nn.train(X, y)

# Make predictions
predictions = nn.predict(X)
accuracy = np.mean(predictions == y) * 100
print(f'Accuracy: {accuracy}%')