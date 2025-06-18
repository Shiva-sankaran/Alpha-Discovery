use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyReadonlyArray1};
use std::f64;

#[pyfunction]
fn compute_all_factors(
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;

    let len = close.len();
    let mut result: Py<PyDict> = Python::with_gil(|py| PyDict::new(py).into());


    macro_rules! add_array {
        ($name:expr, $data:expr) => {
            Python::with_gil(|py| {
                result
                    .as_ref(py)
                    .set_item($name, PyArray1::from_vec(py, $data))
                    .unwrap();
            });
        };
    }

    // 1. Momentum (n-day return)
    for &period in &[5, 10, 20] {
        let mut m = vec![f64::NAN; len];
        for i in period..len {
            if close[i - period] != 0.0 {
                m[i] = close[i] / close[i - period] - 1.0;
            }
        }
        add_array!(format!("momentum_{}", period), m);
    }

    // 2. RSI
    for &period in &[5, 10, 20] {
        let mut rsi = vec![f64::NAN; len];
        let mut gains = vec![0.0; len];
        let mut losses = vec![0.0; len];

        for i in 1..len {
            let delta = close[i] - close[i - 1];
            if delta > 0.0 {
                gains[i] = delta;
            } else {
                losses[i] = -delta;
            }
        }

        for i in period..len {
            let avg_gain: f64 = gains[i - period + 1..=i].iter().sum::<f64>() / period as f64;
            let avg_loss: f64 = losses[i - period + 1..=i].iter().sum::<f64>() / period as f64;

            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        add_array!(format!("rsi_{}", period), rsi);
    }

    // 3. Daily returns
    let mut returns = vec![f64::NAN; len];
    for i in 1..len {
        if close[i - 1] != 0.0 {
            returns[i] = close[i] / close[i - 1] - 1.0;
        }
    }

    // 4. Volatility
    for &period in &[5, 10, 20] {
        let mut vol = vec![f64::NAN; len];
        for i in period..len {
            let slice = &returns[i - period + 1..=i];
            let mean = slice.iter().copied().sum::<f64>() / period as f64;
            let std = slice
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / period as f64;
            vol[i] = std.sqrt();
        }
        add_array!(format!("volatility_{}", period), vol);
    }

    // 5. Volume ratio
    for &period in &[10, 20] {
        let mut ratio = vec![f64::NAN; len];
        for i in period..len {
            let ma = volume[i - period + 1..=i].iter().sum::<f64>() / period as f64;
            if ma != 0.0 {
                ratio[i] = volume[i] / ma;
            }
        }
        add_array!(format!("volume_ma_ratio_{}", period), ratio);
    }

    // 6. Price-volume trend
    let mut pvt = vec![f64::NAN; len];
    for i in 1..len {
        if close[i - 1] != 0.0 {
            pvt[i] = (close[i] - close[i - 1]) / close[i - 1] * volume[i];
        }
    }
    add_array!("price_volume_trend", pvt);

    // 7. Mean reversion z-score
    for &period in &[10, 20, 50] {
        let mut mr = vec![f64::NAN; len];
        for i in period..len {
            let slice = &close[i - period + 1..=i];
            let mean = slice.iter().copied().sum::<f64>() / period as f64;
            let std = slice
                .iter()
                .map(|c| (c - mean).powi(2))
                .sum::<f64>()
                / period as f64;
            let std = std.sqrt();
            if std > 1e-8 {
                mr[i] = (close[i] - mean) / std;
            }
        }
        add_array!(format!("mean_reversion_{}", period), mr);
    }

    Ok(result)
}
#[pymodule]
fn rust_factors(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_all_factors, m)?)?;
    Ok(())
}
