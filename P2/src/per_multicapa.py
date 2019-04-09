# -*- coding: utf-8 -*-
import numpy as np
from redNeuronal import RedNeuronal
import matplotlib.pyplot as plt

class PerceptronMulticapa(RedNeuronal):

    def sigmoide_bipolar(self, x):
        return 2/(1+np.exp(-x))-1

    def dsigmoide_bipolar(self, fx):
        # parametro fx indica el valor de la funcion sigmoide para no recalcular
        # exponenciales
        return (1-fx)*(1+fx)/2

    def __init__(self, num_input, num_output, ncapa, alpha = 0.1, max_epocas = 100):
        super().__init__(num_input, num_output, ncapa, self.sigmoide_bipolar, alpha, max_epocas,self.random_init)

    def random_init(self, x, y):
        return np.random.random_sample((x,y)) - 0.5

    def propagacion(self, dato):
        _in = []
        x = dato
        for capa in self.capas:
            x = np.concatenate(([1], x))
            x = capa.activacion(x)
            _in.append(x)
        return _in


    def train(self, X_train, y_train):
        contador = 0
        #np.place(X_train, X_train==0, -1)
        #np.place(y_train, y_train==0, -1)
        while contador < self.max_epocas: #condicion de parada
            for dato in range(X_train.shape[0]):
                # Propagacion
                matriz=[]
                capas =  self.propagacion(X_train[dato])
                capas.insert(0, X_train[dato])
                y = capas[-1]
                delta = (y_train[dato] - y)*self.dsigmoide_bipolar(y)
                act = np.concatenate(([1],capas[-2]))
                correccion = self.alpha*delta.reshape(-1,1)*act
                matriz.append(correccion)
                for capa in range(len(capas)-2):
                    z = capas[-2-capa]
                    delta_in = np.dot(delta,self.capas[-1-capa].weights[:,1:])
                    delta = delta_in*self.dsigmoide_bipolar(z)
                    act = np.concatenate(([1],capas[-3-capa]))
                    correccion = self.alpha*delta.reshape(-1,1)*act
                    matriz.insert(0,correccion)

                for i in range(len(matriz)):
                    self.capas[i].weights+=matriz[i]
            contador+=1



if __name__ == "__main__":
    from particionado import Modo1
    for f in ["problema_real5"]:
        print("----------"+f+"----------")
        datos = Modo1("../data/"+f+".txt", 0.8, normalizacion="estandar")
        error=[]
        ecm=[]
        for i in [[2], [2,2], [2,2,2], [2,2,2,2], [2,2,2,2,2]]:
            ada = PerceptronMulticapa(datos.X_train.shape[1],datos.y_train.shape[1], i)
            ada.train(datos.X_train, datos.y_train)
            prediction = ada.predict(datos.X_test)
            ecm.append(np.mean(ada.ecm(datos.X_test, datos.y_test)))
            #ada.matriz(datos.y_test, prediction)
            error.append(np.mean(ada.precision(datos.y_test, prediction)))

        plt.plot(error)
        plt.xlabel('Numero de capas ocultas')
        plt.title('Precision en funcion del numero de capas ocultas (con 2 neuronas cada una)')
        plt.figure()
        plt.plot(ecm)
        plt.xlabel('Numero de capas ocultas')
        plt.title('Error cuadratico medio en funcion del numero de capas ocultas (con 2 neuronas cada una)')
        plt.show()
