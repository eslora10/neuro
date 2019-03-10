# -*- coding: utf-8 -*-
import numpy as np

class Capa():
    def __init__(self, num_input, num_output, umbral):
        self.weights = np.zeros((num_output, num_input+1)) #Input + 1 por el bias
        self.umbral = umbral

    def activacion(self, input_value):
        salida = []
        for i in range(self.weights.shape[0]):
            y_in = np.dot(input_value, self.weights[i,:])
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

    def ecm(self, X_test, y_test):
        D = y_test.shape[1]
        N = y_test.shape[0]
        err = [0, 0]
        for i in range(N):
            x = np.concatenate(([1], X_test[i]))
            for capa in self.capas[:-1]:
                x = capa.activacion(x)
            for j in range(D):
                y_in = np.dot(x, self.capas[-1].weights[j,:])
                err[j] += (y_test[i][j]-y_in)**2

        return [i/(2*N) for i in err]

    def precision(self, y_test, prediction):
        err = y_test != prediction
        return sum(err)/y_test.shape[0]


    def matriz(self, y_test, prediction):
        tp=0
        fp=0
        tn=0
        fn=0
        for i in range(y_test.shape[0]):
            if np.array_equal(y_test[i],prediction[i]):
                if np.array_equal(y_test[i],[1,0]):
                    tn+=1
                else:
                    tp+=1
            else:
                if np.array_equal(y_test[i],[1,0]):
                    fn+=1
                else:
                    fp+=1
        print("\t\tMATRIZ DE CONFUSION")
        print("\t\tValor predicho")
        print("\t\tbenigno\tmaligno")
        print("\tbenigno\t{}\t\t{}".format(tn,fn))
        print("Valor real")
        print("\tmaligno\t{}\t\t{}".format(fp,tp))


