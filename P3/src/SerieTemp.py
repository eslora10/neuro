# -*- coding: utf-8 -*-
import argparse
import numpy as np
from Crear_alfabeto import escribe_fichero
from particionado import Temporal
from per_multicapa import PerceptronMulticapa
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Entrena y evalua la prediccion de series temporales')
parser.add_argument('fich_entrada', help='fichero con la serie en el formato correcto')
parser.add_argument('capas', type=int, metavar='ncapas', nargs='+', help='neuronas de las capas intermedias')
parser.add_argument('--tam_train', type=int, default=100, help="porcentaje de datos en train")
parser.add_argument('--alpha', type=float, default=0.1, help="constante de aprendizaje")
parser.add_argument('--nepocas', type=int, default=100, help="numero de epocas de entrenamiento")
parser.add_argument('--tol', type=float, default=0, help="tolerancia para el entrenamiento")
parser.add_argument('--npred', type=int, default=0, help="numero de predicciones recursivas")
parser.add_argument('--plot', type=bool, default=False, help="muestra una grafica de salida")
args = parser.parse_args()

# Definicion de la funcion de activacion y su derivada
def activacion_lineal(x):
    return x

def dactivacion_lineal(x):
    return 1

def ecm_basico(X_test, y_test):
    D = y_test.shape[1]
    N = y_test.shape[0]
    err = [0 for _ in range(D)]
    for i in range(N):
        x = [X_test[i][-1] for _ in range(D)]
        for j in range(D):
            err[j] += (y_test[i][j]-x[j])**2

    return np.mean([i/(2*N) for i in err])

# Particionado del fichero
datos = Temporal(args.fich_entrada, args.tam_train)

# Entrenamiento
red = PerceptronMulticapa(datos.X_train.shape[1], datos.y_train.shape[1], args.capas, args.alpha, args.nepocas, args.tol,
                          fact_salida = activacion_lineal, dfact_salida = dactivacion_lineal)
red.train(datos.X_train, datos.y_train)

# Test
prediction = red.predict(datos.X_test)

# Prediccion recursivas
x = datos.X_test[-1]
y = datos.y_test[-1]
rec = []
cont = 0
while cont < args.npred:
    x = np.concatenate((x, y))[-datos.num_atributos:]
    y = red.predict(np.array([x]))[0]
    rec+=list(x)
    cont+=len(x)
    #print(x, y)
rec+=list(y)

print("ECM en test: {}".format(red.ecm(datos.X_test, datos.y_test)))
print("ECM basico: {}".format(ecm_basico(datos.X_test, datos.y_test)))

if args.plot:
    teorico =[]
    for dato in datos.X_train:
        teorico.append(dato[0])
    for dato in datos.X_test:
        teorico.append(dato[0])
    teorico += list(datos.y_test[-1])
    predicho = []
    for dato in prediction:
        predicho.append(dato[0])
    predicho+=list(prediction[-1])
    t = range(len(teorico))
    plt.plot(t, teorico, label = "valor teorico")
    plt.plot(t[-len(predicho):], predicho, color='r', label = "valor predicho")
    if args.npred != 0:
        plt.plot(range(len(t), len(t)+len(rec)), rec, label = "prediccion recursiva")
    plt.xlabel("t")
    plt.ylabel("valor de la serie")
    plt.legend()
    plt.title("Na={0}, Ns={1}".format(datos.num_atributos, datos.num_clases))
    plt.savefig("../graficas/Na{0}Ns{1}Ep{2}Train{3}.jpg".format(
        datos.num_atributos, datos.num_clases, args.nepocas,
        args.tam_train), format = "jpg")
    plt.show()
