#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <span>

constexpr size_t N = 512;
constexpr size_t BLOCK_SIZE = 32;
constexpr size_t N_RED = N / BLOCK_SIZE;

__global__ void algo(float *data, float *data_out) {
  int block_offset = blockIdx.x * blockDim.x;
  int idx = block_offset + threadIdx.x;
  int idx_local = threadIdx.x;

  if (idx < N) {
    __shared__ float buffer[BLOCK_SIZE];
    buffer[idx_local] = data[idx];

    __syncthreads();

    int stride = 2;
    while (stride <= BLOCK_SIZE) {
      int prev_stride = stride / 2;
      if (idx_local % stride == 0) {
        buffer[idx_local] += buffer[idx_local + prev_stride];
      }
      stride *= 2;
      __syncthreads();
    }

    if (idx_local == 0) {
      data_out[blockIdx.x] = buffer[0];
    }
  }
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 10.0);

  float *dev_data, *dev_data_out;
  cudaMallocHost(&dev_data, N * sizeof(float));
  cudaMallocHost(&dev_data_out, N_RED * sizeof(float));
  auto hst_data = std::span<float>(dev_data, dev_data + N);
  auto hst_data_out = std::span<float>(dev_data_out, dev_data_out + N_RED);
  std::generate(std::begin(hst_data), std::end(hst_data), [&]() { return dis(gen); });

  algo<<<N_RED, BLOCK_SIZE>>>(dev_data, dev_data_out);
  cudaDeviceSynchronize();

  auto dev_sum = std::accumulate(std::begin(hst_data_out), std::end(hst_data_out), 0.0);
  auto sum = std::accumulate(std::begin(hst_data), std::end(hst_data), 0.0);

  std::cout << dev_sum << std::endl;
  std::cout << sum << std::endl;

  cudaFree(dev_data);
  cudaFree(dev_data_out);

  return 0;
}