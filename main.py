from neural_network import NeuralNetwork, normalize, to_one_hot, accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
data = pd.read_csv('breastfinal.csv')

X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 

print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
print(np.min(Xtrain), np.max(Xtrain), np.mean(Xtrain), np.std(Xtrain))
#normalize features if nonbinary dataset

#Xtrain = normalize(Xtrain)  # or use normalize(X_train) if you want zero mean
#Xtest = normalize(Xtest)

numclasses = len(np.unique(y))
unique_labels = np.unique(y)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
ytrain = np.array([label_to_int[label] for label in ytrain])
ytest = np.array([label_to_int[label] for label in ytest])

ytrainoh = to_one_hot(ytrain, numclasses)
ytestoh = to_one_hot(ytest, numclasses)

#building the neural network
#Xtrain.shape[1]  # Number of features
#consider expanding hidden layers to 32, 32 or 64, 64 for more complex datasets
hiddenlayer1 = 16  #number of neurons first hidden layer
hiddenlayer2 = 16  #number of neurons in second hidden layer
nn = NeuralNetwork([Xtrain.shape[1], hiddenlayer1, hiddenlayer2, numclasses], ['relu', 'relu', 'softmax'])
nn.learn(Xtrain, ytrainoh, epochs=100, learning_rate=0.001, batch_size=32)
predictions = nn.predict(Xtest)
print(f"Test Accuracy: {accuracy(ytestoh, predictions) * 100:.2f}%")

'''
Current dataset is rather small, with the current batch size of 32, batches may have high variance which leads to overfitting.
'''