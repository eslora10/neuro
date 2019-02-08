import numpy as np

class Neurona():

    def __init__(self, pesos, umbral):
        self.pesos = pesos
        self.umbral = umbral

    def activacion(self, entrada):
        return np.dot(self.pesos, entrada) >= self.umbral


if __name__ == "__main__":

    n = Neurona(np.array([1,1]), 2)
    print(n.activacion(np.array([0,0])))
    print(n.activacion(np.array([0,1])))
    print(n.activacion(np.array([1,0])))
    print(n.activacion(np.array([1,1])))
