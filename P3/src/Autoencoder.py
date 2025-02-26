# -*- coding: utf-8 -*-
import argparse
import numpy as np
from Crear_alfabeto import escribe_fichero
from particionado import Modo3
from per_multicapa import PerceptronMulticapa


parser = argparse.ArgumentParser(description='Entrena y evalua el autoencoder de letras')
parser.add_argument('fich_entrada', help='fichero alfabeto.dat o subconjunto')
parser.add_argument('capas', type=int, metavar='ncapas', nargs='+', help='neuronas de las capas intermedias')
parser.add_argument('--num_copias_train', default = 1, help='define cuantas veces cada letra es creada como dato en train', type=int)
parser.add_argument('--num_copias_test', default = 1, help='define cuantas veces cada letra es creada como dato en train', type=int)
parser.add_argument('--num_errores_train', default = 0, help='numero de pixeles con errores en train', type=int)
parser.add_argument('--num_errores_test', default = 0, help='numero de pixeles con errores en test', type=int)
parser.add_argument('--alpha', type=float, default=0.1, help="constante de aprendizaje")
parser.add_argument('--nepocas', type=int, default=100, help="numero de epocas de entrenamiento")
parser.add_argument('--tol', type=float, default=0, help="tolerancia para el entrenamiento")
args = parser.parse_args()

# Generar los conjuntos de datos de train y test
escribe_fichero(args.fich_entrada, "letras_train.txt", args.num_errores_train, args.num_copias_train)
escribe_fichero(args.fich_entrada, "letras_test.txt", args.num_errores_test, args.num_copias_test)

# Particionado con modo 3
datos = Modo3("letras_train.txt", "letras_test.txt", None)

# Conversion de los datos a bipolar
np.place(datos.X_train, datos.X_train == 0, -1)
np.place(datos.y_train, datos.y_train == 0, -1)
np.place(datos.X_test, datos.X_test == 0, -1)

# Entrenamiento
red = PerceptronMulticapa(datos.X_train.shape[1], datos.y_train.shape[1], args.capas, args.alpha, args.nepocas, args.tol)
red.train(datos.X_train, datos.y_train)

# Test
prediction = red.predict(datos.X_test)
np.place(prediction, prediction >= 0, 1)
np.place(prediction, prediction < 0, 0)

# Evaluacion
np.place(datos.y_test, datos.y_test == 0, -1)
np.place(prediction, prediction == 0, -1)
pe = 0
lrc = 0
cont = 0
for i in range(len(prediction)):
    pe += red.pixeles_errados(datos.y_test[i], prediction[i])
    lrc += red.letras_recuperadas_correctamente(datos.y_test[i], prediction[i])
    cont+=1

print("Pixeles errados: {0}".format(pe))
print("Media pixeles errados: {0}".format(pe/cont))
print("Letras recuperadas correctamente: {0}".format(lrc))
