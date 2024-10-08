#include <iostream>
#include <vector>
#include <algorithm>
#include <valarray>

constexpr size_t N_BLOCKS = 64;
constexpr size_t N_THREADS = 256;

__global__ void add_vec(double* a, double* b, double* res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N_BLOCKS * N_THREADS) {
        res[idx] = a[idx] + b[idx];
    }
}

int main() {
    constexpr size_t N = N_BLOCKS * N_THREADS;
    auto hst_a = std::vector<double>(N, 1.0);
    auto hst_b = std::vector<double>(N, 2.0);
    auto hst_res = std::vector<double>(N, 0.0);

    double *dev_a, *dev_b, *dev_res;
    cudaMalloc(&dev_a, N * sizeof(double));
    cudaMalloc(&dev_b, N * sizeof(double));
    cudaMalloc(&dev_res, N * sizeof(double));

    cudaMemcpy(dev_a, hst_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, hst_b.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    add_vec<<<N_BLOCKS, N_THREADS>>>(dev_a, dev_b, dev_res);
    cudaDeviceSynchronize();

    cudaMemcpy(hst_res.data(), dev_res, N * sizeof(double), cudaMemcpyDeviceToHost);

    // validate
    auto val_res = std::valarray<double>(hst_res.data(), N);
    auto val_ab = std::valarray<double>(hst_a.data(), N) + std::valarray<double>(hst_b.data(), N);

    std::cout << std::ranges::equal(val_res, val_ab) << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    return 0;
}