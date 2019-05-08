all:

autoencoder_help:
	python3 ./src/Autoencoder.py -h

autoencoder_exec:
	python3 ./src/Autoencoder.py ./data/alfabeto_dat.txt 9 --num_copias_train 10 --num_copias_test 4 --num_errores_train 10 --num_errores_test 0
serie_help:
	python3 ./src/SerieTemp.py -h

serie_exec:
	@echo "Generando serie1 con Na=5, Ns=1..."
	python3 ./src/Adapta_fichero_serie.py ./data/p3_serie1.txt ./data/serie1.txt 5 1
	@echo "Entrenando red neuronal..."
	python3 ./src/SerieTemp.py ./data/serie1.txt 10 --nepocas 1000 --npred 0 --tam_train 100
