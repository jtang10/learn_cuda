#include <cstddef>
#include <vector>
#include <iostream>
#include <chrono>

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
__kernel__ gpu_baseline_sgemm(float *C, const float *A, const float *B,
  const size_t M, const size_t N, const size_t K)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    // if out of bound of matrix C
    if ((indexX >= M) || (indexY >= N)
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

int main()
{
  int M = 1000;
  int N = 1000;
  int K = 500;
  std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

  vector<float> A(M*K);
  vector<float> B(K*N);
  vector<float> C(M*N);

  std::chrono::high_resolution_clock Clock;

  for (int i = 0; i < A.size(); i++)
    A[i] = i;
  for (int i = 0; i < B.size(); i++)
    B[i] = i;

  auto start = Clock.now();
  cpu_sgemm(C.data(), A.data(), B.data(), M, N, K);
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(Clock.now() - start);
  std::cout << "Time spent on cpu_sgemm: " << duration.count() << " ms\n";

  // std::cout << "C" << "\n";
  // for (int i = 0; i < M; i++)
  // {
  //   for (int j = 0; j < N; j++)
  //   {
  //     std::cout << C[i*N + j] << ' ';
  //   }
  //   std::cout << "\n";
  // }
}