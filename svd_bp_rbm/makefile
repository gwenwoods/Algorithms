all: RBM_RUN SVD_RUN BP_RUN

RBM_RUN: RBM_main.o abstractAlgorithm.o rbm.o
	g++ -O3 RBM_main.o abstractAlgorithm.o rbm.o -o RBM_RUN -lm

RBM_main.o: RBM_main.cpp
	g++ -c RBM_main.cpp

SVD_RUN: SVD_main.o abstractAlgorithm.o svd.o
	g++ -O3 SVD_main.o abstractAlgorithm.o svd.o -o SVD_RUN -lm

SVD_main.o: SVD_main.cpp
	g++ -c SVD_main.cpp

abstractAlgorithm.o: abstractAlgorithm.cpp
	g++ -c abstractAlgorithm.cpp


rbm.o: rbm.cpp
	g++ -c rbm.cpp

BP_RUN: BP_main.o abstractAlgorithm.o bp.o
	g++ -O3 BP_main.o abstractAlgorithm.o bp.o -o BP_RUN -lm

BP_main.o: BP_main.cpp
	g++ -c BP_main.cpp

bp.o: bp.cpp
	g++ -c bp.cpp

clean:
	rm -rf *o *RUN

