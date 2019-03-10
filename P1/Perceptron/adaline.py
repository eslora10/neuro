# -*- coding: utf-8 -*-
import numpy as np
from redNeuronal import RedNeuronal

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
                    try:
                        w = w+delta
                    except RuntimeWarning:
                        print(w)
                    self.capas[0].weights[j] = w
                    if max_val < np.max(abs(delta)):
                        max_val = np.max(abs(delta))
            err = max_val
            nepocas += 1
        print("Entrenado en {0} epocas".format(nepocas))
        if nepocas == self.max_epocas:
            print("Se ha alcanzado el maximo numero de epocas.")
        else:
            print("La variacion de los pesos es menor que la tolerancia")

    def predict(self, X_test):
        np.place(X_test, X_test == 0, -1)
        pred = super().predict(X_test)
        np.place(pred, pred == 0, 1)
        np.place(pred, pred == -1, 0)
        return pred

if __name__ == "__main__":
    import particionado as p

    datos = p.Modo1("./data/problema_real1.txt", 0.7)

    neuron = Adaline(0.3, 0.2, datos.X_train.shape[1], datos.y_train.shape[1])
    neuron.train(datos.X_train, datos.y_train)
    prediction = neuron.predict(datos.X_test)
    neuron.matriz(datos.y_test,prediction)
    print("Porcentaje de error al clasificar los datos: " + str(neuron.precision(datos.y_test, prediction)))
    print("Error cuadrático medio al clasificar los datos: " + str(neuron.ecm(datos.X_test, datos.y_test)))
