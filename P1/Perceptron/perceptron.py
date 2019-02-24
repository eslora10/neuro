# -*- coding: utf-8 -*-
import numpy as np



class Perceptron():

    def __init__(self, alpha, umbral, tamentrada, tamsalida):
        self.alpha = alpha
        self.umbral = umbral
        self.pesos=np.zeros((tamsalida,tamentrada+1)) #inicializamos pesos y sesgo a 0 (una fila por neurona de salida, tantas columnas como neuronas de entrada + el sesgo)


    def entrenamiento(self,datostrain, clasestrain):

        contador=1
        while contador ==1 or not np.array_equal(pesos_nueva_epoca,pesos_actual) :
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

    def clasifica(self, datostest, tamsalida):
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

    def errores(self,predicciones, datostest):
        err=0
        
        for i in range(datostest.shape[0]):
            if not np.array_equal(datostest[i], predicciones[i]):
                err+=1
        return err/datostest.shape[0]



                    
        