# -*- coding: utf-8 -*-
import numpy as np
from redNeuronal import RedNeuronal

class PerceptronMulticapa(RedNeuronal):

    def sigmoide_bipolar(x):
        return 2/(1+np.exp(-x))-1

    def dsigmoide_bipolar(fx):
        # parametro fx indica el valor de la funcion sigmoide para no recalcular
        # exponenciales
        return (1-fx)*(1+fx)/2

    def __init__(self, num_input, num_output, ncapa, alpha = 0.1, max_epocas = 100, init_pesos):
        super().__init__(num_input, num_output, ncapa, sigmoide_bipolar, alpha, max_epocas, init_pesos)

    def random_init(x, y):
        return np.random.random_sample((x,y)) - 0.5

    def propagacion(dato):
        _in = []
        x = np.concatenate(([1], dato))
        for capa in self.capas:
            x = capa.activacion(x)
            _in.append(x)
        return _in


    def train(self, X_train, y_train):
        while contador < self.max_epocas: #condicion de parada
            for dato in range(X_train.shape[0]):
                # Propagacion
                _in = self.predict([X_train[dato]])[0]
                for k in range(y.shape[1]):
                    delta = (y_train[dato][k] - y[k])*dsigmoide_bipolar(y[k])
