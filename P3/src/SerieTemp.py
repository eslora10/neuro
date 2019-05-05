# -*- coding: utf-8 -*-
import argparse
import numpy as np
from Crear_alfabeto import escribe_fichero
from particionado import Temporal
from per_multicapa import PerceptronMulticapa

parser = argparse.ArgumentParser(description='Entrena y evalua la prediccion de series temporales')
parser.add_argument('fich_entrada', help='fichero con la serie en el formato correcto')
parser.add_argument('capas', type=int, metavar='ncapas', nargs='+', help='neuronas de las capas intermedias')
parser.add_argument('--tam_train', type=float, default=0.7, help="porcentaje de datos en train")
parser.add_argument('--alpha', type=float, default=0.1, help="constante de aprendizaje")
parser.add_argument('--nepocas', type=int, default=100, help="numero de epocas de entrenamiento")
parser.add_argument('--tol', type=float, default=0, help="tolerancia para el entrenamiento")
parser.add_argument('--npred', type=int, default=0, help="numero de predicciones recursivas")
args = parser.parse_args()

# Definicion de la funcion de activacion y su derivada
def activacion_lineal(x):
    return x

def dactivacion_lineal(x):
    return 1

# Particionado del fichero
datos = Temporal(args.fich_entrada, args.tam_train)

# Entrenamiento
red = PerceptronMulticapa(datos.X_train.shape[1], datos.y_train.shape[1], args.capas, args.alpha, args.nepocas, args.tol,
                          fact_salida = activacion_lineal, dfact_salida = dactivacion_lineal)
red.train(datos.X_train, datos.y_train)

# Test
prediction = red.predict(datos.X_test)
for i in range(5):
    print(datos.y_test[i], prediction[i])

# Prediccion recursivas
x = datos.X_test[-1]
y = datos.y_test[-1]
print(args.npred)
for _ in range(args.npred):
    x = np.concatenate((x, y))[-datos.num_atributos:]
    y = red.predict(np.array([x]))[0]
    print(x, y)
