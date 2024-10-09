#include <xtensor/xhistogram.hpp>
#include <xtensor/xtensor.hpp>

xt::xtensor<float, 1> host_histogram(const xt::xtensor<float, 1> &a, float max, size_t n_bins) {
  return xt::histogram(a, n_bins, 0.0f, max);
}
