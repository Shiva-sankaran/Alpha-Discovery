// strategy.cpp
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

std::vector<double> compute_returns_cpp(
    const std::vector<double>& predictions,
    const std::vector<double>& actual_returns,
    double target_annual_vol,
    double cost_per_unit,
    double& out_leverage,
    std::vector<double>& turnover_cost) {
    
    size_t n = predictions.size();
    std::vector<double> ranks(n);
    std::iota(ranks.begin(), ranks.end(), 0);

    // Rank predictions
    std::sort(ranks.begin(), ranks.end(), [&](int i, int j) {
        return predictions[i] < predictions[j];
    });

    std::vector<double> weights(n);
    for (size_t i = 0; i < n; ++i)
        weights[ranks[i]] = (double(i) - (n + 1.0) / 2.0) / (n / 2.0);  // [-1, 1]

    std::vector<double> gross_returns(n);
    for (size_t i = 0; i < n; ++i)
        gross_returns[i] = weights[i] * actual_returns[i];

    // Compute daily vol
    double mean = std::accumulate(gross_returns.begin(), gross_returns.end(), 0.0) / n;
    double sq_sum = 0.0;
    for (double r : gross_returns) sq_sum += (r - mean) * (r - mean);
    double daily_vol = std::sqrt(sq_sum / n);
    
    double target_daily_vol = target_annual_vol / std::sqrt(252.0);
    out_leverage = (daily_vol > 1e-8) ? (target_daily_vol / daily_vol) : 1.0;

    // Apply leverage and compute net returns
    std::vector<double> net_returns(n), scaled_weights(n);
    turnover_cost.resize(n, 0.0);
    scaled_weights[0] = weights[0] * out_leverage;

    for (size_t i = 0; i < n; ++i) {
        scaled_weights[i] = weights[i] * out_leverage;
        net_returns[i] = scaled_weights[i] * actual_returns[i];
        if (i > 0)
            turnover_cost[i] = cost_per_unit * std::abs(scaled_weights[i] - scaled_weights[i - 1]);
        net_returns[i] -= turnover_cost[i];
    }

    return net_returns;
}
