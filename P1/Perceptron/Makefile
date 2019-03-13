all: mp_exec

per_help:
	@echo "Ejecucion algoritmo perceptron: (python3 main.py -h para ayuda general)"
	@echo "python3 main.py per nombre_del_fichero_entrenamiento modo [--in_file2 nombre_del_fichero_test] [--out_file nombre_del_fichero_salida] [--tam_train porcentaje_train] [--umbral umbral] [--alpha alpha] [--nepocas max_epocas_entrenamiento]"

per_exec:
	python3 main.py per data/problema_real2.txt 1 --umbral 0.5 --alpha 0.1 --tam_train 0.8 --nepocas 500


ada_help:
	@echo "Ejecucion algoritmo adaline: (python3 main.py -h para ayuda general)"
	@echo "python3 main.py ada nombre_del_fichero_entrenamiento modo [--in_file2 nombre_del_fichero_test] [--out_file nombre_del_fichero_salida] [--tam_train porcentaje_train] [--tol tol] [--alpha alpha] [--nepocas max_epocas_entrenamiento]"

ada_exec:
	python3 main.py ada data/problema_real2.txt 1 --tol 0.1 --alpha 0.1

per_and:
	python3 main.py per data/and.txt 2 --umbral 1 --alpha 0.1

ada_or:
	python3 main.py ada data/or.txt 2 --tol 0.1 --alpha 0.1

per_modo3:
	python3 main.py per data/problema_real2.txt 3 --in_file2 data/problema_real2_no_etiquetados.txt --out_file predicciones/prediccion_perceptron.txt --umbral 1 --alpha 0.1

ada_modo3:
	python3 main.py ada data/problema_real2.txt 3 --in_file2 data/problema_real2_no_etiquetados.txt --out_file predicciones/prediccion_adaline.txt --tol 0.1 --alpha 0.1

make: