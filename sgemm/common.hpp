#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#ifdef __CUDACC__
inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s@%d: CUDA Runtime Error(%d): %s\n", file, line,
            int(result), cudaGetErrorString(result));
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
#endif

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

template <typename T>
int all_close(std::vector<T> expected, std::vector<T> actual, float eps)
{
  int size1 = expected.size();
  int size2 = actual.size();
  if (size1 != size2)
  {
    std::cout << "Expected has different size with actual: " << size1
      << " vs. " << size2 << std::endl;
    return -1;
  }

  int count = 0;
  for (size_t i = 0; i < size1; i++)
  {
    count += (abs(expected[i] - actual[i]) > eps);
  }

  return count;
}

template <typename T>
void print_matrix(std::vector<T> matrix, int row, int col)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      std::cout << matrix[i*col + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

inline int random_int() { return (std::rand() % 10); }
