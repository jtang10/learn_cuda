#include <cstddef>
#include <vector>
#include <iostream>
#include <chrono>
#include <argparse/argparse.hpp>
#include "common.hpp"

using std::vector;

/**
 * @brief CPU version of sgemm
 *
 * @param C resulting matrix (M, N)
 * @param A multiplicant matrix (M, K)
 * @param B multiplier matrix (K, N)
 * @param M number of row in C
 * @param N number of column in C
 * @param K number of row in B or column in A
 */
__global__ void gpu_baseline_sgemm(float *C, const float *A, const float *B,
  const int M, const int N, const int K)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    // if out of bound of matrix C
    if ((row >= M) || (col >= N))
      return;

    // each thread responsible for getting its own elements of A & B directly
    // from global memory
    float acc = 0.0;
    for (int i = 0; i < K; i++)
    {
      acc += A[row*K + i] * B[i*K + col];
    }
    C[row*N + col] = acc;
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser program("cpu_sgemm");
  program.add_argument("M")
    .help("row dimension of resulting matrix C")
    .scan<'i', int>();
  program.add_argument("N")
    .help("col dimension of resulting matrix C")
    .scan<'i', int>();
  program.add_argument("K")
    .help("inner dimension of matrix A & B")
    .scan<'i', int>();
  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  auto M = program.get<int>("M");
  auto N = program.get<int>("N");
  auto K = program.get<int>("K");
  std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

  vector<float> A(M*K);
  vector<float> B(K*N);
  vector<float> C(M*N);
  vector<float> CActual(M*N);

  std::chrono::high_resolution_clock Clock;

  std::generate(A.begin(), A.end(), random_int);
  std::generate(B.begin(), B.end(), random_int);
  std::generate(C.begin(), C.end(), random_int);

  // cudaMalloc
  float *dA, *dB, *dC;
  CUDA_RUNTIME(cudaMalloc(&dA, M*K*sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&dB, K*N*sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&dC, M*K*sizeof(float)));

  // cudaMemcpy
  CUDA_RUNTIME(cudaMemcpy(dA, A.data(), M*K*sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaMemcpy(dB, B.data(), K*N*sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaMemcpy(dC, C.data(), M*N*sizeof(float), cudaMemcpyDefault));

  // GPU kernel launch parameters
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  dimGrid.x = (N + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (M + dimBlock.y - 1) / dimBlock.y;

  auto start = Clock.now();
  gpu_baseline_sgemm<<<dimGrid, dimBlock>>>(dC, dA, dB, M, N, K);
  cudaDeviceSynchronize();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(Clock.now() - start);
  std::cout << "Time spent on gpu_baseline_sgemm: " << duration.count() << " ms\n";

  // cudaMemcpy
  CUDA_RUNTIME(cudaMemcpy(CActual.data(), dC, M*N*sizeof(float), cudaMemcpyDefault));

  start = Clock.now();
  cpu_sgemm(C.data(), A.data(), B.data(), M, N, K);
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(Clock.now() - start);
  std::cout << "Time spent on cpu_sgemm: " << duration.count() << " ms\n";

  int num_diff = all_close(C, CActual, 1e-6);
  if (num_diff)
    std::cout << "There are " << num_diff << " out of " << M*N << " not matching" << std::endl;

  // std::cout << "Expected" << std::endl;
  // print_matrix(C, M, N);
  // std::cout << "Actual" << std::endl;
  // print_matrix(CActual, M, N);
}