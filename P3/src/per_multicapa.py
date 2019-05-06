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

    def __init__(self, num_input, num_output, ncapa, alpha = 0.1, max_epocas = 100, tol = 0, plot = False,
                 fact_salida = None, dfact_salida = None):
        self.tol = tol
        self.errores = []
        self.media_errores= []
        self.lrc = []
        self.plot = plot
        if dfact_salida == None:
            self.dfact_salida = self.dsigmoide_bipolar
        else:
            self.dfact_salida = dfact_salida
        super().__init__(num_input, num_output, ncapa, self.sigmoide_bipolar, fact_salida, alpha, max_epocas,self.random_init)

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
                delta = (y_train[dato] - y)*self.dfact_salida(y)
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


            if self.plot:
                #Para mostrar la grafica de la evolucion de las metricas pedidas en train
                self.errores.append(self.pixeles_errados(y_train, y))
                self.media_errores.append(self.pixeles_errados(y_train, y)/y_train.shape[0])
                self.lrc.append(self.letras_recuperadas_correctamente(y_train, y))

            if self.tol != 0:
                # Se especifica una tolerancia, calculamos el ecm sobre train
                # y vemos si es inferior
                err =  self.ecm(X_train, y_train)
                if err < self.tol:
                    print("Se ha alcanzado un valor de ecm menor que la tolerancia")
                    break
        print("Entrenado en {0} epocas".format(contador))
        if self.plot:
            self.plot_pixeles_errados()
            self.plot_pixeles_errados_media()
            self.plot_letras()


    def plot_pixeles_errados(self):
        plt.figure()
        plt.plot(range(len(self.errores)), self.errores)
        plt.ylabel("Pixeles errados")
        plt.xlabel("Epoca de entrenamiento")
        plt.title("Evolucion de los pixeles errados totales en train")
        plt.savefig("pixerrados-epoca.jpg")
        plt.close()

    def plot_pixeles_errados_media(self):
        plt.figure()
        plt.plot(range(len(self.media_errores)), self.media_errores)
        plt.ylabel("Media de pixeles errados")
        plt.xlabel("Epoca de entrenamiento")
        plt.title("Evolucion de la media de pixeles errados por letra en train")
        plt.savefig("media-pixerrados-epoca.jpg")
        plt.close()

    def plot_letras(self):
        plt.figure()
        plt.plot(range(len(self.lrc)), self.lrc)
        plt.ylabel("Letras recuperadas correctamente")
        plt.xlabel("Epoca de entrenamiento")
        plt.title("Evolucion de las letras recuperadas correctamente en train")
        plt.savefig("lrc-epoca.jpg")
        plt.close()