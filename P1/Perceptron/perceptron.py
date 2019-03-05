# -*- coding: utf-8 -*-
import numpy as np
from redNeuronal import RedNeuronal

"""
class Perceptron(RedNeuronal):

    def __init__(self, alpha, umbral, tamentrada, tamsalida):
        self.alpha= alpha
        self.umbral = umbral
        self.pesos=np.zeros((tamsalida,tamentrada+1)) #inicializamos pesos y sesgo a 0 (una fila por neurona de salida, tantas columnas como neuronas de entrada + el sesgo)


    def train(self,datostrain, clasestrain):
        np.place(clasestrain,clasestrain==0,-1)
        print (clasestrain)
        contador=1
        pesos_nueva_epoca=self.pesos
        pesos_actual=np.ones_like(self.pesos)
        while contador <=100 and not np.array_equal(pesos_nueva_epoca,pesos_actual) :
            pesos_nueva_epoca=self.pesos.copy()
            print ("EPOCA " + str(contador) + "\nPesos al inicio: " + str(pesos_nueva_epoca))
            #para cada dato de entrenamiento
            for dato in range(datostrain.shape[0]):
                pesos_actual= self.pesos
                entrada = np.append(datostrain[dato],1) #establecemos activaciones de las neuronas de entrada (el ultimo es para el sesgo)
                salida=[] #guardaremos la y de cada neurona de salida
                for i in range(clasestrain.shape[1]):
                    y_in = np.dot(entrada, pesos_actual[i,:]) #y_in de cada neurona de salida
                    if y_in < -self.umbral:
                        salida.append(-1)
                    elif y_in>self.umbral:
                        salida.append(1)
                    else:
                        salida.append(0)
                #ajustamos pesos y sesgo
                for i in range(len(salida)):
                    if salida[i] != clasestrain[dato,i]:
                        self.pesos[i]= pesos_actual[i] + np.dot(self.alpha*clasestrain[dato,i], entrada)

            print ("Pesos al final: "+ str(self.pesos))
            pesos_actual=self.pesos
            contador +=1

    def predict(self, datostest, tamsalida):
        predicciones=[]

        for dato in range(datostest.shape[0]):
            entrada = np.append(datostest[dato],1) #establecemos activaciones de las neuronas de entrada (el ultimo es para el sesgo)
            prediccion=[]
            for i in range(tamsalida):
                y_in = np.dot(entrada, self.pesos[i,:]) #y_in de cada neurona de salida
                if y_in < -self.umbral:
                    prediccion.append(-1)
                elif y_in>self.umbral:
                    prediccion.append(1)
                else:
                    prediccion.append(0)
            predicciones.append(prediccion)
        return np.array(predicciones)

    def test(self,predicciones, datostest):
        err=0
        np.place(datostest,datostest==0, -1)
        for i in range(datostest.shape[0]):
            if not np.array_equal(datostest[i], predicciones[i]):
                err+=1
        return err/datostest.shape[0]

"""
class Perceptron(RedNeuronal):
    def __init__(self, alpha, umbral, tamentrada, tamsalida, max_epocas=100):
        super().__init__(tamentrada, tamsalida,[], alpha, umbral, max_epocas)


    def train(self,datostrain, clasestrain):
        np.place(clasestrain,clasestrain==0,-1)
        contador=1
        pesos_nueva_epoca=self.capas[0].weights
        pesos_actual=np.ones_like(self.capas[0].weights)
        while contador <=100 and not np.array_equal(pesos_nueva_epoca,pesos_actual) :
            pesos_nueva_epoca=self.capas[0].weights.copy()
            print ("EPOCA " + str(contador) + "\nPesos al inicio: " + str(pesos_nueva_epoca))
            #para cada dato de entrenamiento
            for dato in range(datostrain.shape[0]):
                pesos_actual= self.capas[0].weights
                entrada = np.append(1,datostrain[dato]) #establecemos activaciones de las neuronas de entrada (el primero es para el sesgo)
                salida=[] #guardaremos la y de cada neurona de salida
                for i in range(clasestrain.shape[1]):
                    y_in = np.dot(entrada, pesos_actual[i,:]) #y_in de cada neurona de salida
                    if y_in < -self.umbral:
                        salida.append(-1)
                    elif y_in>self.umbral:
                        salida.append(1)
                    else:
                        salida.append(0)
                #ajustamos pesos y sesgo
                for i in range(len(salida)):
                    if salida[i] != clasestrain[dato,i]:
                        self.capas[0].weights[i]= pesos_actual[i] + np.dot(self.alpha*clasestrain[dato,i], entrada)

            print ("Pesos al final: "+ str(self.capas[0].weights))
            pesos_actual=self.capas[0].weights
            contador +=1

    def predict(self, datostest):
        pred = super().predict(datostest)
        #np.place(pred, pred == 0, 1)
        np.place(pred, pred == -1, 0)
        return pred


if __name__ == "__main__":
    from particionado import *

    modo = Modo1("./data/problema_real1.txt", 0.8)

    p = Perceptron(0.3,3,modo.X_train.shape[1], modo.y_train.shape[1],100)
    p.train(modo.X_train, modo.y_train)

    predicciones = p.predict(modo.X_test)

    print("Porcentaje de error al clasificar los datos: " + str(p.precision(modo.y_test, predicciones)))
    print("Error cuadrático medio al clasificar los datos: " + str(p.ecm(modo.y_test, predicciones)))
