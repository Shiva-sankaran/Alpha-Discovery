import requests

# 1. Register signal
register_payload = {
    "signal_name": "my_dynamic_signal",
    "function_code": """
def signal(df):
    return df["momentum_10"] - df["rsi_10"]
"""
}
resp = requests.post("http://localhost:8000/register_signal/", json=register_payload)
print("Register response:", resp.status_code, resp.json())

# 2. Backtest it
backtest_payload = {
    "symbol": "AAPL",
    "signal_name": "my_dynamic_signal"
}
resp = requests.post("http://localhost:8000/backtest/", json=backtest_payload)
print("Backtest results:", resp.status_code)
print(resp.json())
