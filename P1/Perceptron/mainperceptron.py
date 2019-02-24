# -*- coding: utf-8 -*-
import numpy as np
from particionado import *
from perceptron import Perceptron


modo= Modo1("./data/problema_real1.txt", 0.8)

p= Perceptron(0.5,15,modo.X_train.shape[1], modo.y_train.shape[1])
p.entrenamiento(modo.X_train, modo.y_train)
predicciones=p.clasifica(modo.X_test, modo.y_test.shape[1])
print ("El porcentaje de error en la clasificacion es: " + str(p.errores(predicciones, modo.y_test)))
