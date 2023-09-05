#include <cstddef>
#include <vector>
#include <iostream>
#include <chrono>
// #include <argparse/argparse.hpp>

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
  // argparse::ArgumentParser program("cpu_sgemm");
  // program.add_argument("M")
  //   .help("row dimension of resulting matrix C")
  //   .scan<'i', int>();
  // program.add_argument("N")
  //   .help("col dimension of resulting matrix C")
  //   .scan<'i', int>();
  // program.add_argument("M")
  //   .help("inner dimension of matrix A & B")
  //   .scan<'i', int>();
  // try {
  //   program.parse_args(argc, argv);
  // }
  // catch (const std::runtime_error& err) {
  //   std::cerr << err.what() << std::endl;
  //   std::cerr << program;
  //   return 1;
  // }

  // auto M = program.get<int>("M");
  // auto N = program.get<int>("N");
  // auto K = program.get<int>("K");
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