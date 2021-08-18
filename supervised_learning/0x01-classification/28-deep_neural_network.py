#!/usr/bin/env python3
"""a deep neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """a deep neural network performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """nx is the number of input features to the neuron
        layers is a list representing the number of nodes in each layer"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                a = nx
            else:
                a = layers[i - 1]
            self.__weights['W' + str(i + 1)
                           ] = np.random.randn(layers[i], a) * np.sqrt(2 / a)
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        X: numpy.ndarray with shape (nx, m) that contains the input data"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            z = np.matmul(self.__weights["W" + str(i + 1)],
                          self.__cache["A" + str(i)]) +\
                self.__weights["b" + str(i + 1)]
            if i != self.__L - 1:
                if self.activation == 'sig':
                    self.__cache['A' + str(i + 1)] = self.sig(z)
                elif self.activation == 'tanh':
                    self.__cache['A' + str(i + 1)] = self.tanh(z)
            else:
                self.__cache['A' + str(i + 1)] = np.exp(z) / \
                    np.sum(np.exp(z), axis=0)
        return self.cache['A' + str(self.L)], self.__cache

    @staticmethod
    def sig(x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        """tanh function"""
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray containing the correct labels for the input data
        A is a numpy.ndarray containing the activated output"""
        m = Y.shape[1]
        cost = 1 / m * np.sum(-(Y * np.log(A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions
        X is a numpy.ndarray tcontaining the input data
        Y is a numpy.ndarray containing the correct labels"""
        pred1 = self.forward_prop(X)[0]
        predic2 = np.where(pred1 == np.amax(pred1, axis=0), 1, 0)
        return predic2, self.cost(Y, pred1)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calculates gradient descent
          alpha is the learning rate"""
        (nx, m) = np.shape(Y)
        grad = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            grada = np.matmul(grad, cache["A" + str(i - 1)].T) / m
            gradb = np.sum(grad, axis=1, keepdims=True) / m
            if self.activation == 'sig':
                grad = np.matmul(self.__weights["W" + str(i)].T, grad) * (
                    cache["A" + str(i - 1)] * (1 - cache["A" + str(i - 1)]))
            elif self.activation == 'tanh':
                grad = np.matmul(self.__weights["W" + str(i)].T, grad) * (
                    1 - np.power(cache["A" + str(i - 1)], 2))
            self.__weights["W" + str(i)] -= alpha * grada
            self.__weights["b" + str(i)] -= alpha * gradb

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """Trains the network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            elif step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        x = []
        y = []
        for i in range(iterations):
            A, c = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if verbose and (i < 1 or i % step == 0):
                print(
                    "Cost after {} iterations: {}".format(
                        i, self.cost(
                            Y, A)))
                x.append(self.cost(Y, A))
                y.append(i + step)
        if graph:
            plt.plot(y, x, color="blue")
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename:
            return None
        if not (filename.endswith(".pkl")):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
