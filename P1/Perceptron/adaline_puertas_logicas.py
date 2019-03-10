# -*- coding: utf-8 -*-
from adaline import Adaline
from particionado import Modo2

for f in ["and","nand", "nor", "xor"]:
    print("----------"+f+"----------")
    datos = Modo2("data/"+f+".txt")
    ada = Adaline(0.1, 0.1, 2, 2)
    ada.train(datos.X_train, datos.y_train)
    prediction = ada.predict(datos.X_test)
    print(ada.precision(datos.y_test, prediction))
    print(ada.ecm(datos.X_test, datos.y_test))


