#include <cstddef>
#include <vector>
#include <iostream>

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
void cpu_sgemm(float *C, const float *A, const float *B,
  const size_t M, const size_t N, const size_t K)
{
  for (size_t m = 0; m < M; m++)
  {
    for (size_t n = 0; n < N; n++)
    {
      float acc = 0.0;
      for (size_t k = 0; k < K; k++)
      {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m*N + n] = acc;
    }
  }
}

int main()
{
  int M = 2, N = 4, K = 3;
  vector<float> A(M*K);
  vector<float> B(K*N);
  vector<float> C(M*N);

  for (int i = 0; i < A.size(); i++)
    A[i] = i;
  for (int i = 0; i < B.size(); i++)
    B[i] = i;

  std::cout << "A" << "\n";
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < K; j++)
    {
      std::cout << A[i*K + j] << ' ';
    }
    std::cout << "\n";
  }

  std::cout << "B" << "\n";
  for (int i = 0; i < K; i++)
  {
    for (int j = 0; j < N; j++)
    {
      std::cout << B[i*N + j] << ' ';
    }
    std::cout << "\n";
  }

  cpu_sgemm(C.data(), A.data(), B.data(), M, N, K);
  std::cout << "C" << "\n";
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      std::cout << C[i*N + j] << ' ';
    }
    std::cout << "\n";
  }

}