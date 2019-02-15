# -*- coding: utf-8 -*-
import numpy as np

class Neurona():

    def __init__(self, umbral):
        self.umbral = umbral

    def activacion(self, pesos, entrada):
        return int(np.dot(pesos, entrada) >= self.umbral)

class NeuronaEntrada(Neurona):

    def __init__(self):
        self.umbral = 0

    def activacion(self, pesos, entrada):
        return entrada


if __name__ == "__main__":

    n = Neurona(np.array([1,1]), 2)
    print(n.activacion(np.array([0,0])))
    print(n.activacion(np.array([0,1])))
    print(n.activacion(np.array([1,0])))
    print(n.activacion(np.array([1,1])))
