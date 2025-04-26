#import mnist
from neural_network import NeuralNetwork, normalize, to_one_hot, accuracy

X_train = mnist.train_images()
y_train = mnist.train_labels()
X_test = mnist.test_images()
y_test = mnist.test_labels()

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

y_train_oh = to_one_hot(y_train, 10)
y_test_oh = to_one_hot(y_test, 10)

nn = NeuralNetwork([784, 16, 16, 10], ['relu', 'relu', 'softmax'])
nn.learn(X_train, y_train_oh, epochs=30, learning_rate=0.01, batch_size=32)

predictions = nn.predict(X_test)
print(f"Test Accuracy: {accuracy(y_test_oh, predictions) * 100:.2f}%")