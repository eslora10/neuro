# -*- coding: utf-8 -*-
import numpy as np
from redNeuronal import RedNeuronal
"""
class Adaline():

    def __init__(self, alpha, tol, num_input, num_output, max_epocas = 100):
        self.weights = np.zeros((num_output, num_input+1)) #Input + 1 por el bias
        self.alpha = alpha
        self.tol = tol
        self.max_epocas = max_epocas

    def train(self, X_train, y_train):
        # transformar a bipolar
        np.place(X_train, X_train == 0, -1)
        np.place(y_train, y_train == 0, -1)
        D = y_train.shape[1]
        N = X_train.shape[0]
        err = 99
        nepocas = 0
        while err > self.tol and nepocas < self.max_epocas:
            max_val= 0
            for i in range(N):
                for j in range(D):
                    w = self.weights[j]
                    x = np.concatenate(([1],X_train[i]))
                    t = y_train[i][j]
                    delta = self.alpha*(t-np.dot(w,x))*x
                    w = w+delta
                    self.weights[j] = w
                    if max_val < np.max(abs(delta)):
                        max_val = np.max(abs(delta))
            err = max_val
            nepocas += 1
        print("Entrenado en {0} epocas".format(nepocas))

    def predict(self, X_test):
        np.place(X_test, X_test == 0, -1)
        pred = []
        for i in range(X_test.shape[0]):
            x = np.concatenate(([1], X_test[i]))
            y_in = np.dot(self.weights, x)
            pred.append([1 if y >= 0 else 0 for y in y_in])
        return np.array(pred)

def test(y_test, prediction):
    np.place(y_test, y_test == -1, 0)
    err = y_test != prediction
    return sum(err)/y_test.shape[0]
"""

class Adaline(RedNeuronal):

    def __init__(self, alpha, tol, num_input, num_output, max_epocas = 100):
        super().__init__(num_input, num_output,[], alpha = alpha, max_epocas = max_epocas)
        self.tol = tol

    def train(self, X_train, y_train):
        # transformar a bipolar
        np.place(X_train, X_train == 0, -1)
        np.place(y_train, y_train == 0, -1)
        D = y_train.shape[1]
        N = X_train.shape[0]
        err = 99
        nepocas = 0
        while err > self.tol and nepocas < self.max_epocas:
            max_val= 0
            for i in range(N):
                for j in range(D):
                    w = self.capas[0].weights[j]
                    x = np.concatenate(([1],X_train[i]))
                    t = y_train[i][j]
                    delta = self.alpha*(t-np.dot(w,x))*x
                    w = w+delta
                    self.capas[0].weights[j] = w
                    if max_val < np.max(abs(delta)):
                        max_val = np.max(abs(delta))
            err = max_val
            nepocas += 1
        print("Entrenado en {0} epocas".format(nepocas))

    def predict(self, X_test):
        pred = super().predict(X_test)
        np.place(pred, pred == 0, 1)
        np.place(pred, pred == -1, 0)
        return pred

if __name__ == "__main__":
    import particionado as p

    datos = p.Modo1("./data/problema_real1.txt", 0.8)

    neuron = Adaline(0.1, 0.2, datos.X_train.shape[1], datos.y_train.shape[1])
    neuron.train(datos.X_train, datos.y_train)
    prediction = neuron.predict(datos.X_test)
    print(neuron.precision(datos.y_test, prediction))
    print(neuron.ecm(datos.y_test, prediction))
