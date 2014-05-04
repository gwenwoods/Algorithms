all: SVD_RUN

SVD_RUN: SVD_main.o abstractAlgorithm.o svd.o
	g++ -O3 SVD_main.o abstractAlgorithm.o svd.o -o SVD_RUN -lm

SVD_main.o: SVD_main.cpp
	g++ -c SVD_main.cpp

abstractAlgorithm.o: abstractAlgorithm.cpp
	g++ -c abstractAlgorithm.cpp


svd.o: svd.cpp
	g++ -c svd.cpp

clean:
	rm -rf *o SVD_RUN

