#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

void cudaCheckError(const char *f, int l) {
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		fprintf(stderr, "CudaError:%s:%d: '%s'\n", f, l, cudaGetErrorString(err));
		exit(-1);
	}
}

#define cuda(name, ...) (cuda##name(__VA_ARGS__), cudaCheckError(__func__, __LINE__))

__global__ void GPU_Kernel() {
  printf(" GPU block  : %d / %d  GPU thread : %d / %d\n",
         blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}


int main(int argc, char** argv) {

  //MPI & GPU init
  int mpisize = -1, mpirank = -1, gpusize = -1, gpurank = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

  cuda(GetDeviceCount, &gpusize);
  cuda(SetDevice, mpirank % gpusize);
  cuda(GetDevice, &gpurank);

/*
行列A，B，Cを用意
各rankに対し，sizeで分割された小さな行列を作る．
MPIの受信用にrecvを用意
*/
  constexpr size_t N = 2048;
  constexpr size_t matrix_size = N * N * sizeof(float);
  float (*A)[N], (*B)[N], (*C)[N];
  A = (float(*)[N])malloc(matrix_size);
  B = (float(*)[N])malloc(matrix_size);
  C = (float(*)[N])malloc(matrix_size);

  //subA, ..., recv : float[][N]
  float (*subA)[N], (*subB)[N], (*subC)[N], (*recv)[N];
  cuda(MallocManaged, &subA, matrix_size/mpisize);
  cuda(MallocManaged, &subB, matrix_size/mpisize);
  cuda(MallocManaged, &subC, matrix_size/mpisize);

  cuda(MallocManaged, &recv, matrix_size/mpisize);

  // random na kazu de syokika
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }

  //Aを小さな行列にコピー
  int offset = N/mpisize*mpirank;
  for (int i=0; i<N/mpisize; i++)
    for (int j=0; j<N; j++)
      subA[i][j] = A[(i+offset)][j];

  //Bを小さな行列にコピー. 転置する
  for (int i=0; i<N; i++)
    for (int j=0; j<N/mpisize; j++)
      subB[j][i] = B[i][j+offset];
      //subB[N/mpisize*i+j] = B[i][j+offset];

  //ring 通信のアドレス
  int recv_from = (mpirank + 1) % mpisize;
  int send_to = (mpirank - 1 + mpisize) % mpisize;

  double comp_time = 0, comm_time = 0;
  for(int impirank=0; impirank<mpisize; impirank++) {

    auto tic = chrono::steady_clock::now();

    offset = N/mpisize*((mpirank+impirank) % mpisize);

    //行列席の計算．結果は部分的な行列
    //size回のループでsubCが埋まる．一回で１マス埋まる
    for (int i=0; i<N/mpisize; i++)
      for (int j=0; j<N/mpisize; j++)
        for (int k=0; k<N; k++)
          subC[i][j+offset] += subA[i][k] * subB[j][k];

    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();

    //ring通信
    MPI_Request request[2];
    MPI_Isend(&subB[0], N*N/mpisize, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&recv[0], N*N/mpisize, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);

    //行列Bを新しい行列に更新
    for (int i=0; i<N/mpisize; i++)
      for(int j = 0; j < N; j++)
        subB[i][j] = recv[i][j];

    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();

  }

  //結果を集める
  MPI_Allgather(&subC[0], N*N/mpisize, MPI_FLOAT, &C[0], N*N/mpisize, MPI_FLOAT, MPI_COMM_WORLD);

  //err keisoku
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[i][j] -= A[i][k] * B[k][j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[i][j]);
  // kokomade

  // syutsuryoku
  if(mpirank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n", time, 2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  MPI_Finalize();
}
