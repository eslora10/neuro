# -*- coding: utf-8 -*-
import numpy as np

class Capa():
    def __init__(self, num_input, num_output, umbral):
        self.weights = np.zeros((num_output, num_input+1)) #Input + 1 por el bias
        self.umbral = umbral

    def activacion(self, input):
        salida = []
        for i in range(self.weights.shape[0]):
            y_in = np.dot(input, self.weights[i,:])
            if y_in < -self.umbral:
                salida.append(-1)
            elif y_in>self.umbral:
                salida.append(1)
            else:
                salida.append(0)
        return np.array(salida)

class RedNeuronal():
    def __init__(self, num_input, num_output, ncapa, alpha = 0.1, umbral = 0, max_epocas = 100):
        self.umbral = umbral
        self.max_epocas = max_epocas
        self.alpha = alpha
        self.capas = []
        n = num_input
        for num_neuronas in ncapa:
            self.capas.append(Capa(n, num_neuronas, umbral))
            n = num_neuronas
        self.capas.append(Capa(n, num_output, umbral))

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pred = []
        for i in range(X_test.shape[0]):
            x = np.concatenate(([1], X_test[i]))
            for capa in self.capas:
                x = capa.activacion(x)
            pred.append(x)

        return np.array(pred)

    def ecm(self, y_test, prediction):
        D = y_test.shape[1]
        N = y_test.shape[0]
        return np.array([np.linalg.norm(y_test[:,i]-prediction[:,i])**2 for i in range(D)])

    def precision(self, y_test, prediction):
        err = y_test != prediction
        return sum(err)/y_test.shape[0]
