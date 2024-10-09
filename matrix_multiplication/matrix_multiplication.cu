#include <algorithm>
#include <chrono>
#include <cuda/barrier>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>

constexpr size_t BLOCKS = 512;
constexpr size_t N = 32;
using namespace std::chrono_literals;

xt::xtensor<double, 2> matmul(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b);

__global__ void kern_matmul(double *a, double *b, double *res) {
  int offset = blockIdx.x * N * N;
  int row = threadIdx.y;
  int col = threadIdx.x;

  if (row < N && col < N) {
    // cuda::barrier<cuda::thread_scope_block> block(N*N);

    __shared__ float a_shared[N][N + 1];
    __shared__ float b_shared[N][N + 1];

    a_shared[row][col] = a[offset + row * N + col];
    b_shared[row][col] = b[offset + row * N + col];

    // block.arrive_and_wait();
    __syncthreads();

    float sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += a_shared[row][i] * b_shared[i][col];
    }

    res[offset + row * N + col] = sum;

    // block.arrive_and_wait();
    __syncthreads();
  }
}

int main() {
  auto hst_a = std::array<std::vector<double>, BLOCKS>{};
  auto hst_b = std::array<std::vector<double>, BLOCKS>{};
  auto hst_res = std::array<std::vector<double>, BLOCKS>{};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 10.0);

  double *dev_a, *dev_b, *dev_res;
  cudaMalloc(&dev_a, BLOCKS * N * N * sizeof(double));
  cudaMalloc(&dev_b, BLOCKS * N * N * sizeof(double));
  cudaMalloc(&dev_res, BLOCKS * N * N * sizeof(double));

  for (size_t i = 0; i < BLOCKS; ++i) {
    hst_a[i] = std::vector<double>(N * N, 0.0);
    hst_b[i] = std::vector<double>(N * N, 0.0);
    hst_res[i] = std::vector<double>(N * N, 0.0);
    std::generate(hst_a[i].begin(), hst_a[i].end(), [&]() { return dis(gen); });
    std::generate(hst_b[i].begin(), hst_b[i].end(), [&]() { return dis(gen); });

    cudaMemcpy(dev_a + i * N * N, hst_a[i].data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b + i * N * N, hst_b[i].data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  }

  auto start = std::chrono::high_resolution_clock::now();
  constexpr size_t N_ITER = 100;
  for (int i = 0; i < N_ITER; i++) {
    dim3 matrix_size(N, N);
    kern_matmul<<<BLOCKS, matrix_size>>>(dev_a, dev_b, dev_res);
    cudaDeviceSynchronize();
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / N_ITER;
  std::cout << "Time taken by function: " << duration << " microseconds" << std::endl;

  // validate
  try {
    for (size_t i = 0; i < BLOCKS; ++i) {
      cudaMemcpy(hst_res[i].data(), dev_res + i * N * N, N * N * sizeof(double), cudaMemcpyDeviceToHost);
      auto val_res = xt::adapt(hst_res[i], {N, N});
      auto val_a = xt::adapt(hst_a[i], {N, N});
      auto val_b = xt::adapt(hst_b[i], {N, N});

      auto val_ab = matmul(val_a, val_b);

      if (!xt::allclose(val_res, val_ab)) {
        throw std::runtime_error("Validation failed");
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_res);

  return 0;
}