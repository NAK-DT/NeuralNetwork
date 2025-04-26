import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self.learning_rate = None

        #weights and biases initialization
        for i in range(len(layers) - 1):
            in_size = layers[i]
            out_size = layers[i + 1]
            limit = np.sqrt(6 / (in_size + out_size))
            w = np.random.uniform(-limit, limit, (out_size, in_size))
            b = np.zeros(out_size)
            self.weights.append(w)
            self.biases.append(b)

    def _activation(self, x, func):
        if func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'relu':
            return np.maximum(0, x)
        elif func == 'leaky_relu':
            return np.where(x > 0, x, x * 0.01)
        elif func == 'linear':
            return x
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {func}")

    def _activation_derivative(self, x, func):
        if func == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif func == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif func == 'relu':
            return (x > 0).astype(float)
        elif func == 'leaky_relu':
            return np.where(x > 0, 1.0, 0.01)
        elif func == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function for derivative: {func}")

    def _cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def _cross_entropy_derivative(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]

    def feedforward(self, X):
        a = X
        activations = [a]
        zs = []

        for w, b, func in zip(self.weights, self.biases, self.activations):
            z = a @ w.T + b
            a = self._activation(z, func)
            zs.append(z)
            activations.append(a)

        return activations, zs

    def backpropagate(self, X, y, activations, zs, loss_function='cross_entropy_loss'):
        delta = self._cross_entropy_derivative(y, activations[-1])

        nablaw = [np.zeros_like(w) for w in self.weights]
        nablab = [np.zeros_like(b) for b in self.biases]

        for l in reversed(range(len(self.weights))):
            nablaw[l] = delta.T @ activations[l]
            nablab[l] = np.sum(delta, axis=0)

            if l != 0:
                delta = (delta @ self.weights[l]) * self._activation_derivative(zs[l-1], self.activations[l-1])

        return nablaw, nablab

    def updateWeights(self, nabla_w, nabla_b, batch_size):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * nabla_w[i] / batch_size
            self.biases[i] -= self.learning_rate * nabla_b[i] / batch_size

    def learn(self, X_train, y_train, epochs, learning_rate, batch_size, verbose=True):
        self.learning_rate = learning_rate

        for epoch in range(1, epochs + 1):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]

                activations, zs = self.feedforward(batch_X)
                nabla_w, nabla_b = self.backpropagate(batch_X, batch_y, activations, zs)
                self.updateWeights(nabla_w, nabla_b, batch_X.shape[0])

            if verbose and epoch % 5 == 0:
                preds, _ = self.feedforward(X_train)
                loss = self._cross_entropy_loss(y_train, preds[-1])
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

    def predict(self, X):
        activations, _ = self.feedforward(X)
        return activations[-1]

def to_one_hot(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def normalize(X):
    X = X.astype(np.float32)
    X -= X.mean(axis=0)
    X /= (X.std(axis=0) + 1e-8)
    return X

def accuracy(y_true, y_pred):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(true_labels == pred_labels)
