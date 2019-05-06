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
            self.num_atributos = int(params[0])
            self.num_clases = int(params[1])
            for line in infile:
                params = line.strip("\n").split()
                self.atr.append([float(i) for i in params[:self.num_atributos]])
                self.clases.append([float(i) for i in params[self.num_atributos:]])
            infile.close()

        self.atr = np.array(self.atr)
        self.clases = np.array(self.clases)


class Modo1(Particionado):

    def __init__(self, file, tam_train, normalizacion = ""):
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

        if normalizacion == "estandar":
            medias = np.mean(self.X_train, axis=0)
            sd = np.std(self.X_train, axis=0)
            for i in range(self.X_train.shape[0]):
                self.X_train[i]= np.divide(self.X_train[i]-medias,sd)
            for i in range(self.X_test.shape[0]):
                self.X_test[i]= np.divide(self.X_test[i]-medias,sd)

        elif normalizacion == "min-max":
            min = self.X_train.min(axis=0)
            max = self.X_train.max(axis=0)
            for i in range(self.X_train.shape[0]):
                self.X_train[i]= 2*np.divide(self.X_train[i]-min, max-min)-1
            for i in range(self.X_test.shape[0]):
                self.X_test[i]= 2*np.divide(self.X_test[i]-min, max-min)-1

class Modo2(Particionado):

    def __init__(self, file, normalizacion= ""):
        super().__init__(file)
        self.X_train = self.atr
        self.y_train = self.clases
        self.X_test = deepcopy(self.atr)
        self.y_test = deepcopy(self.clases)
        if normalizacion == "estandar":
            medias = np.mean(self.X_train, axis=0)
            sd = np.std(self.X_train, axis=0)
            for i in range(self.X_train.shape[0]):
                self.X_train[i]= np.divide(self.X_train[i]-medias,sd)
            for i in range(self.X_test.shape[0]):
                self.X_test[i]= np.divide(self.X_test[i]-medias,sd)

        elif normalizacion == "min-max":
            min = self.X_train.min(axis=0)
            max = self.X_train.max(axis=0)
            for i in range(self.X_train.shape[0]):
                self.X_train[i]= 2*np.divide(self.X_train[i]-min, max-min)-1
            for i in range(self.X_test.shape[0]):
                self.X_test[i]= 2*np.divide(self.X_test[i]-min, max-min)-1


class Modo3(Particionado):

    def __init__(self, file_train, file_test, normalizacion= ""):
        super().__init__(file_train)
        self.X_train = self.atr
        self.y_train = self.clases

        super().__init__(file_test)
        self.X_test = self.atr
        self.y_test = self.clases
        if normalizacion == "estandar":
            medias = np.mean(self.X_train, axis=0)
            sd = np.std(self.X_train, axis=0)
            for i in range(self.X_train.shape[0]):
                self.X_train[i]= np.divide(self.X_train[i]-medias,sd)
            for i in range(self.X_test.shape[0]):
                self.X_test[i]= np.divide(self.X_test[i]-medias,sd)

        elif normalizacion == "min-max":
            min = self.X_train.min(axis=0)
            max = self.X_train.max(axis=0)
            for i in range(self.X_train.shape[0]):
                self.X_train[i]= 2*np.divide(self.X_train[i]-min, max-min)-1
            for i in range(self.X_test.shape[0]):
                self.X_test[i]= 2*np.divide(self.X_test[i]-min, max-min)-1


class Temporal(Particionado):
    """ Similar al modo 1 pero sin realizar un shuffle de los datos
    """
    def __init__(self, file, tam_train):
        super().__init__(file)
        n_datos = self.atr.shape[0]
        idx = np.array(range(n_datos))
        idx_train = idx[:tam_train]
        idx_test = idx[tam_train:]

        self.X_train = self.atr[idx_train]
        self.y_train = self.clases[idx_train]
        self.X_test = self.atr[idx_test]
        self.y_test = self.clases[idx_test]

