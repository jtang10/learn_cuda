# learn_cuda
all about CUDA programming, profiling & optimization

This repo is inspired and built upon [nvidia-performance-tools](https://github.com/cwpearson/nvidia-performance-tools) but I will continue to expand from there.

## Prerequisite
1. CUDA. My version is 11.7 but should work with any recent versions.
2. Nsight Compute & System. Should ship with CUDA installation.
3. [argparse](https://github.com/p-ranav/argparse): It seems to require gcc 8.1+ since my 7.5 lacks charconv file during compilation.

## Roadmap
### sgemm
- [x] CPU baseline
- [ ] boilerplate for checking correctness and argparse
- [ ] GPU baseline
- [ ] GPU shared memory baseline
- [ ] GPU thread corsening
### reduction
### convolution


