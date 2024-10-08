#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

xt::xtensor<double, 2> matmul(const xt::xtensor<double, 2> &a, const xt::xtensor<double, 2> &b) {
  xt::xtensor<double, 2> ab = xt::zeros<double>({a.shape(0), b.shape(1)});
  for (size_t i = 0; i < a.shape(0); i++) {
    auto row_a = xt::view(a, i, xt::all());
    for (size_t j = 0; j < b.shape(1); j++) {
      auto col_b = xt::view(b, xt::all(), j);
      ab(i, j) = xt::sum(row_a * col_b)();
    }
  }
  return ab;
}