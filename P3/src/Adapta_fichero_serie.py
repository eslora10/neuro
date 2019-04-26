# -*- coding: utf-8 -*-
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Transforma el fichero con la serie temporal al formato aceptado por la red neuronal')
parser.add_argument('fich_entrada', help='fichero con la serie temporal')
parser.add_argument('fich_salida', help='fichero resultado')
parser.add_argument('Na', help='numero de entradas', type=int)
parser.add_argument('Ns', help='numero de salidas', type=int)
args = parser.parse_args()

def split_sequence(sequence, n_x, n_y):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_x
        end_iy = end_ix + n_y
        if end_iy > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_iy]
        X.append(seq_x)
        y.append(seq_y)
    return X, y

sequence = []
with open(args.fich_entrada, "r") as entrada:
    for line in entrada:
        sequence.append(line.strip("\n"))

X, y = split_sequence(sequence, args.Na, args.Ns)

with open(args.fich_salida, "w") as salida:
    salida.write("{0}\t{1}\n".format(args.Na, args.Ns))
    for i in range(len(X)):
        salida.write("\t".join(X[i]+y[i])+"\n")
