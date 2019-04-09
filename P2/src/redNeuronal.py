# -*- coding: utf-8 -*-
import numpy as np

class Capa():
    def __init__(self, num_input, num_output, activacion, pesos):
        self.weights = pesos(num_output, num_input+1) #Input + 1 por el bias
        self.factivacion = activacion

    def activacion(self, input_value):
        salida = []
        for i in range(self.weights.shape[0]):
            y_in = np.dot(input_value, self.weights[i,:])
            salida.append(self.factivacion(y_in))

        return np.array(salida)

class RedNeuronal():
    def __init__(self, num_input, num_output, ncapa, factivacion, alpha = 0.1, max_epocas = 100, pesos = np.zeros):
        self.max_epocas = max_epocas
        self.alpha = alpha
        self.capas = []
        n = num_input
        for num_neuronas in ncapa:
            self.capas.append(Capa(n, num_neuronas, factivacion, pesos))
            n = num_neuronas
        self.capas.append(Capa(n, num_output, factivacion, pesos))

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pred = []
        for i in range(X_test.shape[0]):
            x = X_test[i]
            for capa in self.capas:
                x = np.concatenate(([1], x))
                x = capa.activacion(x)
            pred.append(x)

        return np.array(pred)

    def ecm(self, X_test, y_test):
        D = y_test.shape[1]
        N = y_test.shape[0]
        err = [0, 0]
        for i in range(N):
            x = X_test[i]
            for capa in self.capas:
                x = np.concatenate(([1], x))
                x = capa.activacion(x)

            for j in range(D):
                err[j] += (y_test[i][j]-x[j])**2

        return [i/(2*N) for i in err]

    def precision(self, y_test, prediction):
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                if prediction[i,j]<0.5:
                    prediction[i,j]=0
                else :
                    prediction[i,j]=1
        err = y_test != prediction
        return sum(err)/y_test.shape[0]


    def matriz(self, y_test, prediction):
        tp=0
        fp=0
        tn=0
        fn=0

        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                if prediction[i,j]<0.5:
                    prediction[i,j]=0
                else :
                    prediction[i,j]=1

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
        print("\t\t\tValor real")
        print("\t\tClase positiva\tClase negativa")
        print("\tClase positiva\t{}\t\t{}".format(tp,fp))
        print("Valor predicho")
        print("\tClase negativa\t{}\t\t{}".format(fn,tn))
