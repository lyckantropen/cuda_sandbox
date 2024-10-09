#include <iostream>
#include <random>
#include <span>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

xt::xtensor<float, 1> host_histogram(const xt::xtensor<float, 1> &a, float max, size_t n_bins);

constexpr size_t BLOCKS = 512;
constexpr size_t THREADS = 32;
constexpr size_t N_BINS = 10;
constexpr float MAX = 10.0;

__global__ void histogram(const float *data, size_t n, float *hist, float bin_width) {
  int block_offset = blockIdx.x * blockDim.x;
  int idx = threadIdx.x;

  if (block_offset + idx < n) {
    __shared__ float local_hist[N_BINS];
    if (idx < N_BINS) {
      local_hist[idx] = 0.0;
    }
    __syncthreads();

    int bin = floorf(data[block_offset + idx] / bin_width);
    atomicAdd(&local_hist[bin], 1.0);

    __syncthreads();

    if (idx < N_BINS) {
      atomicAdd(&hist[idx], local_hist[idx]);
    }
  }
  //// global memory version
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx < n) {
  //   int bin = floorf(data[idx] / bin_width);
  //   atomicAdd(&hist[bin], 1.0);
  // }
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, MAX);

  float *dev_data;
  float *dev_hist;
  cudaMallocHost(&dev_data, BLOCKS * THREADS * sizeof(float));
  cudaMallocHost(&dev_hist, N_BINS * sizeof(float));

  auto hst_data = std::span<float>(dev_data, dev_data + BLOCKS * THREADS);
  auto hst_hist = std::span<float>(dev_hist, dev_hist + N_BINS);
  std::generate(hst_data.begin(), hst_data.end(), [&]() { return dis(gen); });
  std::fill(hst_hist.begin(), hst_hist.end(), 0.0);

  auto bin_width = MAX / N_BINS;

  histogram<<<BLOCKS, THREADS>>>(dev_data, BLOCKS * THREADS, dev_hist, 1.0);
  cudaDeviceSynchronize();

  auto hst_hist_xt = xt::adapt(hst_hist.data(), {N_BINS});
  auto hst_data_xt = xt::adapt(hst_data.data(), {BLOCKS * THREADS});

  auto hist_xt = host_histogram(hst_data_xt, MAX, N_BINS);

  if (xt::allclose(hist_xt, hst_hist_xt)) {
    std::cout << "Success!" << std::endl;
  } else {
    std::cout << "Failure!" << std::endl;
  }
}