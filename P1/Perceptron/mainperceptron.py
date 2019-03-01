# -*- coding: utf-8 -*-
import numpy as np
from particionado import *
from perceptron import Perceptron
from pylab import *
import matplotlib.pyplot as plt


modo= Modo1("./data/problema_real1.txt", 0.8)
errores=[]
for i in range (200):
	p= Perceptron(1,i,modo.X_train.shape[1], modo.y_train.shape[1])
	p.train(modo.X_train, modo.y_train)
	predicciones=p.predict(modo.X_test, modo.y_test.shape[1])
	errores.append(p.test(predicciones,modo.y_test))
plt.figure()
plt.plot(errores)
plt.show()

