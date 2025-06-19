# sdk/backtester.py
import pandas as pd
from register_signal import get_signal
from main import AlphaFactorAnalysis




def run_backtest(symbol: str, df: pd.DataFrame, signal_name: str) -> dict:
    signal_func = get_signal(signal_name)
    signal = signal_func(df)

    analyzer = AlphaFactorAnalysis()
    result = analyzer.linear_regression_analysis(
        signal, df["close"].pct_change().shift(-5), signal_name
    )
    print(result)
    return result
