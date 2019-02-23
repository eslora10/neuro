# -*- coding: utf-8 -*-
import numpy as np

class Adaline():

    def __init__(self, num_input, num_output, alpha = 1):
        self.weights = np.zeros((num_output, num_input+1)) #Input + 1 por el bias
        self.alpha = alpha

    def train(self, tol, X_train, y_train):
        # transformar a bipolar
        np.place(X_train, X_train == 0, -1)
        np.place(y_train, y_train == 0, -1)
        D = y_train.shape[1]
        N = X_train.shape[0]
        err = 99
        nepocas = 0
        while err > tol and nepocas < 100:
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

if __name__ == "__main__":
    import particionado as p

    datos = p.Modo1("./data/problema_real1.txt", 0.8)

    neuron = Adaline(datos.X_train.shape[1], datos.y_train.shape[1], alpha = 0.1)
    neuron.train(0.1, datos.X_train, datos.y_train)
    prediction = neuron.predict(datos.X_test)
    print(test(datos.y_test, prediction))
