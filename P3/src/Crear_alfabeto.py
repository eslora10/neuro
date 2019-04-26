# -*- coding: utf-8 -*-
import numpy as np

def escribe_letra(salida, letra, num_errores, num_copias):
    N = len(letra)
    for _ in range(num_copias):
        idx = np.random.randint(0, N, size=num_errores)
        copia = []
        for i in range(N):
            if i not in idx:
                copia.append(letra[i])
            else:
                copia.append(str(1-int(letra[i])))
        salida.write('\t'.join(copia)+'\t'+'\t'.join(letra)+'\n')

def escribe_fichero(fich_entrada, fich_salida, num_errores, num_copias):
    letra = []
    ini = 1
    with open(fich_entrada, 'r') as entrada:
        with open(fich_salida, 'w') as salida:
            for line in entrada:
                if line[0] == '/' and letra:
                    if ini:
                        salida.write('{0}\t{0}\n'.format(len(letra)))
                        ini = 0
                    escribe_letra(salida, letra, num_errores, num_copias)
                    letra = []
                elif line[0] == '0' or line[0] == '1':
                    letra += line.strip('\n').split()
            escribe_letra(salida, letra, num_errores, num_copias)

if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description='Transforma el fichero con el alfabeto al formato aceptado por la red neuronal')
    parser.add_argument('num_copias', help='define cuantas veces cada letra es creada como dato', type=int)
    parser.add_argument('num_errores', help='numero de píxeles con errores en la entrada a la red', type=int)
    parser.add_argument('fich_entrada', help='fichero alfabeto.dat o subconjunto')
    parser.add_argument('fich_salida', help='fichero resultado')

    args = parser.parse_args()
    escribe_fichero(args.fich_entrada, args.fich_salida, args.num_errores, args.num_copias)
