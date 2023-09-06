#include <vector>
#include <iostream>
#include <chrono>
#include <argparse/argparse.hpp>
#include "common.hpp"

using std::vector;


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

  std::chrono::high_resolution_clock Clock;

  std::generate(A.begin(), A.end(), std::rand);
  std::generate(B.begin(), B.end(), std::rand);
  std::generate(C.begin(), C.end(), std::rand);

  auto start = Clock.now();
  cpu_sgemm(C.data(), A.data(), B.data(), M, N, K);
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(Clock.now() - start);
  std::cout << "Time spent on cpu_sgemm: " << duration.count() << " ms\n";
}