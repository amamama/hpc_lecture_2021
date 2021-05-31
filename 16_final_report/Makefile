all: mpi cuda

mpi: a.out

a.out: example.cpp
	mpicxx example.cpp -fopenmp -march=native -O3 -o a.out

cuda: c.out

c.out: example.cu
	nvcc example.cu -lmpi -arch=sm_60 -O3 -Xcompiler "-march=native -O3 -fopenmp" -o c.out

run: all
	qsub -g tga-hpc-lecture job.sh

