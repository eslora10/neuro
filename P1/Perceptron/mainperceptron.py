# -*- coding: utf-8 -*-
import numpy as np
from particionado import *
from perceptron import Perceptron
from pylab import *
import matplotlib.pyplot as plt


modo= Modo1("./data/problema_real2.txt", 0.8)
errores=[]
for i in range (200):
	p = Perceptron(0.1,i,modo.X_train.shape[1], modo.y_train.shape[1],100)
	p.train(modo.X_train, modo.y_train)
	predicciones = p.predict(modo.X_test)
	errores.append(np.mean(p.precision(modo.y_test, predicciones)))
plt.figure()
plt.plot(errores)
plt.show()

