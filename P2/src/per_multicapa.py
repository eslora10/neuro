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

    def __init__(self, num_input, num_output, ncapa, alpha = 0.1, max_epocas = 100):
        super().__init__(num_input, num_output, ncapa, sigmoide_bipolar, alpha, max_epocas)

    def train(self, X_train, y_train):
        pass
