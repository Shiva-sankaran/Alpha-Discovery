// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "strategy.cpp"

namespace py = pybind11;

PYBIND11_MODULE(strategy_cpp, m) {
    m.def("compute_returns_cpp",
          [](const std::vector<double>& preds,
             const std::vector<double>& actuals,
             double target_annual_vol,
             double cost_per_unit) {
              double leverage;
              std::vector<double> turnover;
              auto returns = compute_returns_cpp(preds, actuals, target_annual_vol, cost_per_unit, leverage, turnover);
              return py::make_tuple(returns, leverage, turnover);
          },
          py::arg("predictions"),
          py::arg("actual_returns"),
          py::arg("target_annual_vol") = 0.10,
          py::arg("cost_per_unit") = 0.005);
}