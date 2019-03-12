# -*- coding: utf-8 -*-
import numpy as np
import random
from copy import deepcopy

class Particionado():

    def __init__(self, file):
        self.atr = []
        self.clases = []
        with open(file) as infile:
            l = infile.readline() #num atributos, clases
            params = l.strip("\n").split()
            num_atributos = int(params[0])
            num_clases = int(params[1])
            for line in infile:
                params = line.strip("\n").split()
                self.atr.append([float(i) for i in params[:num_atributos]])
                self.clases.append([float(i) for i in params[num_atributos:]])
            infile.close()

        self.atr = np.array(self.atr)
        self.clases = np.array(self.clases)


class Modo1(Particionado):

    def __init__(self, file, tam_train):
        super().__init__(file)
        n_datos = self.atr.shape[0]
        idx = np.array(range(n_datos))
        np.random.shuffle(idx)
        idx_train = idx[:int(tam_train*n_datos)]
        idx_test = idx[int(tam_train*n_datos):]
        self.X_train = self.atr[idx_train]
        self.y_train = self.clases[idx_train]
        self.X_test = self.atr[idx_test]
        self.y_test = self.clases[idx_test]

class Modo2(Particionado):

    def __init__(self, file):
        super().__init__(file)
        self.X_train = self.atr
        self.y_train = self.clases
        self.X_test = deepcopy(self.atr)
        self.y_test = deepcopy(self.clases)

class Modo3(Particionado):

    def __init__(self, file_train, file_test):
        super().__init__(file_train)
        self.X_train = self.atr
        self.y_train = self.clases

        super().__init__(file_test)
        self.X_test = self.atr
        self.y_test = self.clases


if __name__ == "__main__":
    p = Modo3("./data/problema_real1.txt", "salida.txt")
    print(p.X_train[0:])
    print(p.X_test)
    print(p.y_train.shape[0])
    print(p.y_test)
