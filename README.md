# Student id: 19M30494

やったこと：
行列の転置，MPI+CUDA化
Makefile，jobファイルの追加

キャッシュの授業で行列Bの空間的局所性を上げると良いと習ったので，最初に行列Bを転置するコードを書いた．
事前にMPI化されていた行列Cの部分行列を計算する部分を`06_cuda/10_mpi.cu`，`08_cache_gpu/02_grid.cu`，`08_cache_gpu/03_shared.cu` を参考にしてCUDA化した．
GPUのshared memoryを使用したがMPIの通信がボトルネックで`03_shared.cu`よりも性能は出なかった．
また，Nを倍々にしていくと誤差が4倍ずつ増えていった．これはサイズがNの2乗に比例するからだと考えられる．


# hpc_lecture

|          | Topic                                | Sample code               |
| -------- | ------------------------------------ | ------------------------- |
| Class 1  | Introduction to parallel programming |                           |
| Class 2  | Shared memory parallelization        | 02_openmp                 |
| Class 3  | Distributed memory parallelization   | 03_mpi                    |
| Class 4  | SIMD parallelization                 | 04_simd                   |
| Class 5  | GPU programming 1                    | 05_openacc                |
| Class 6  | GPU programming 2                    | 06_cuda                   |
| Class 7  | Parallel programing models           | 07_starpu                 |
| Class 8  | Cache blocking                       | 08_cache_cpu,08_cache_gpu |
| Class 9  | High Performance Python              | 09_python                 |
| Class 10 | I/O libraries                        | 10_io                     |
| Class 11 | Parallel debugger                    | 11_debugger               |
| Class 12 | Parallel profiler                    | 12_profiler               |
| Class 13 | Containers                           |                           |
| Class 14 | Scientific computing                 | 14_pde                    |
| Class 15 | Deep Learning                        | 15_dl                     |
