# -*- coding: utf-8 -*-

import particionado as p
from adaline import Adaline
import matplotlib.pyplot as plt

datos = p.Modo1("./data/problema_real1.txt", 0.7)

values = [i/100 for i in range(1, 101)]
err =[]
ecm1 = []
ecm2 = []
for alpha in values:
    print(alpha)
    neuron = Adaline(alpha, 0.1, datos.X_train.shape[1], datos.y_train.shape[1])
    neuron.train(datos.X_train, datos.y_train)
    prediction = neuron.predict(datos.X_test)
    err.append(neuron.precision(datos.y_test, prediction)[0])
    ecm = neuron.ecm(datos.X_test, datos.y_test)
    ecm1.append(ecm[0])
    ecm2.append(ecm[1])
fig = plt.figure()
plt.plot(values, err)
plt.plot(values, ecm1)
plt.plot(values, ecm2)
plt.xlabel(r'$\alpha$')
plt.ylabel('error')
plt.savefig("Error_alpha.png")
