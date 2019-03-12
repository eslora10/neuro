# -*- coding: utf-8 -*-
import argparse
import particionado as pt
import adaline as ada
import perceptron as per
from sys import stdout
import numpy as np

parser = argparse.ArgumentParser(description='Ejecuta un algoritmo de red neuronal')
parser.add_argument('alg', help="ada para Adaline, per para Perceptron")
parser.add_argument('in_file', help="fichero de entrada para entenar")
parser.add_argument('--in_file2', help="fichero de entrada para entenar en Modo3")
parser.add_argument('modo', type=int, help="Modo 1, 2 o 3")
parser.add_argument('--out_file')
parser.add_argument('--tam_train', type=float, default=0.7)
parser.add_argument('--umbral', type=float, default=0)
parser.add_argument('--tol', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--nepocas', type=int, default=100)

args = parser.parse_args()
if args.modo == 1:
    datos = pt.Modo1(args.in_file, args.tam_train)
elif args.modo == 2:
    datos = pt.Modo2(args.in_file)
elif args.modo == 3:
    datos = pt.Modo3(args.in_file, args.in_file2)
else:
    parser.print_help()
    exit()

if args.alg == "per":
    red = per.Perceptron(args.alpha, args.umbral, datos.X_train.shape[1], datos.y_train.shape[1],
                         args.nepocas)
elif args.alg =="ada":
    print(args.alpha, args.tol)
    red = ada.Adaline(args.alpha, args.tol, datos.X_train.shape[1], datos.y_train.shape[1],
                      args.nepocas)
else:
    parser.print_help()
    exit()
    
red.train(datos.X_train, datos.y_train)
prediction = red.predict(datos.X_test)

if args.modo==3: #Modo 3 reescribimos el fichero de test con las clases predichas
    np.place(datos.X_test, datos.X_test == -1, 0)
    try:
        if args.out_file:
            f = open(args.out_file, "w")
        else:
            f = open(args.in_file2, "w")
    except:
        f = stdout
    f.write(str(datos.X_test.shape[1]) + "  " + str(prediction.shape[1]) + "\n")
    for i in range(datos.X_test.shape[0]):
        linea= ""
        for j in range(datos.X_test.shape[1]):
            linea += str(datos.X_test[i, j]) +"  "
        for j in range(prediction.shape[1]):
            linea += str(prediction[i][j])+ "  "
        f.write(linea+"\n") 
else: # Modos 1 y 2, nos interesa ver el error al clasificar
    try:
        f = open(args.out_file, "w")

    except:
        f = stdout

    for i in range(prediction.shape[0]):

        for j in range(prediction.shape[1]):
            f.write(str(prediction[i][j])+"  ")
        f.write("\n")
    print("Porcentaje de error al clasificar los datos: " + str(red.precision(datos.y_test, prediction)))
    print("Error cuadrático medio al clasificar los datos: " + str(red.ecm(datos.X_test, datos.y_test)))
