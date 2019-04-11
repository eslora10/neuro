# -*- coding: utf-8 -*-
import argparse
import particionado as pt
from per_multicapa import PerceptronMulticapa
from sys import stdout
import numpy as np

parser = argparse.ArgumentParser(description='Ejecuta el algoritmo del perceptron multicapa')
parser.add_argument('in_file', help="fichero de entrada para entenar")
parser.add_argument('--in_file2', help="fichero de entrada para entenar en Modo3")
parser.add_argument('modo', type=int, help="Modo 1, 2 o 3")
parser.add_argument('capas', type=int, metavar='ncapas', nargs='+', help='neuronas de las capas intermedias')
parser.add_argument('--out_file', help="fichero para las predicciones")
parser.add_argument('--tam_train', type=float, default=0.7, help="porcentaje de datos para entrenamiento")
parser.add_argument('--alpha', type=float, default=0.1, help="constante de aprendizaje")
parser.add_argument('--nepocas', type=int, default=100, help="numero de epocas de entrenamiento")
parser.add_argument('--tol', type=float, default=0, help="tolerancia para el entrenamiento")
parser.add_argument('--norm', default="estandar", help="estandar para normalizacion estandas, min-max para scaling")

args = parser.parse_args()
if args.modo == 1:
    datos = pt.Modo1(args.in_file, args.tam_train, args.norm)
elif args.modo == 2:
    datos = pt.Modo2(args.in_file, args.norm)
elif args.modo == 3:
    datos = pt.Modo3(args.in_file, args.in_file2, args.norm)
else:
    parser.print_help()
    exit()

red = PerceptronMulticapa(datos.X_train.shape[1], datos.y_train.shape[1], args.capas, args.alpha, args.nepocas, args.tol)

red.train(datos.X_train, datos.y_train)
prediction = red.predict(datos.X_test)

if args.modo==3: #Modo 3 reescribimos el fichero de test con las clases predichas
    np.place(datos.X_test, datos.X_test == -1, 0)
    try:
        if args.out_file:
            f = open(args.out_file, "w")
        else:
            f = open(args.in_file2, "w")
        f.write(str(datos.X_test.shape[1]) + " " + str(prediction.shape[1]) + "\n")
        for i in range(datos.X_test.shape[0]):
            linea= ""
            for j in range(datos.X_test.shape[1]):
                linea += str(datos.X_test[i, j]) +" "
            for j in range(prediction.shape[1]):
                if prediction[i][j] >0.5:
                    linea+= str(1) + " "
                else :
                    linea += str(0) + " "
                #linea += str(prediction[i][j])+ " "
            f.write(linea+"\n")
    except:
        pass
else: # Modos 1 y 2, nos interesa ver el error al clasificar
    print("Porcentaje de error al clasificar los datos: " + str(red.precision(datos.y_test, prediction)))
    print("Error cuadrático medio al clasificar los datos: " + str(red.ecm(datos.X_test, datos.y_test)))
    red.matriz(datos.y_test, prediction)
