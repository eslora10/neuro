class Capa():

    neuronas = []

    def __init__(self, neuronas):
        self.neuronas = neuronas

    def addNeurona(self, neurona):
        self.neuronas.append(neurona)

    def activacion(self):
        salida = []
        for neurona in self.neuronas:
            salida.append(neurona.activacion())
        return np.array(salida)

if __name__ == "__main__":
    from neurona import Neurona
    import numpy as np

    x1 = Neurona(valor = 0)
    x2 = Neurona(valor = 1)

    z1 = Neurona(umbral = 2)
    z2 = Neurona(umbral = 2)

    y1 = Neurona(umbral = 2)
    y2 = Neurona(umbral = 2)

    z1.addEntrada(x2, -1)
    z1.addEntrada(z2, 2)
    z2.addEntrada(x2, 2)
    y1.addEntrada(x1, 2)
    y1.addEntrada(z1, 2)
    y2.addEntrada(z2, 2)
    y2.addEntrada(x2, 2)

    capa0 = Capa([x1,x2])
    capa1 = Capa([z2, z1])
    capa3 = Capa([y1, y2])

    print(capa1.activacion())
    x1.valor = 0
    x2.valor = 0

    print(capa3.activacion())
