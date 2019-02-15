# -*- coding: utf-8 -*-
import numpy as np

class Capa():

    def __init__(self):
        self.neuronas = {}

    def addNeurona(self, neurona, pesos):
        self.neuronas[neurona] = pesos

    def activacion(self, entrada):
        salida = []
        for neurona, pesos in self.neuronas.items():
            salida.append(neurona.activacion(pesos, entrada))
        return np.array(salida)

if __name__ == "__main__":
    from neurona import Neurona, NeuronaEntrada

    x1 = NeuronaEntrada()
    x2 = NeuronaEntrada()

    z1 = Neurona(2)
    z2 = Neurona(2)

    y = Neurona(2)

    c1 = Capa()
    c1.addNeurona(x1, None)
    c1.addNeurona(x2, None)

    c2 = Capa()
    c2.addNeurona(z1, np.array([2,-1]))
    c2.addNeurona(z2, np.array([-1,2]))

    c3 = Capa()
    c3.addNeurona(y, np.array([2,2]))


    valores = np.array([x1.activacion(None, np.array(1)), x2.activacion(None, np.array(1))])
    print(valores)
    for capa in [c2, c3]:
        valores = capa.activacion(valores)
    print(valores)
