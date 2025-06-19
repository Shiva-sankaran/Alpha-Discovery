import os
import pandas as pd
import psycopg2
from rust_factors import compute_all_factors  # Rust-Python binding
from datetime import datetime, timedelta

def fetch_ohlcv(conn, symbol: str, days: int = 2000):
    cursor = conn.cursor()
    query = """
        SELECT date, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = %s AND date >= %s
        ORDER BY date ASC;
    """
    start_date = (datetime.today() - timedelta(days=days)).date()
    df = pd.read_sql(query, conn, params=(symbol, start_date))
    df.set_index("date", inplace=True)
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values.astype("float64")
    volume = df["volume"].values.astype("float64")
    factors = compute_all_factors(close, volume)
    return pd.DataFrame(factors, index=df.index)

def save_arrow(df: pd.DataFrame, symbol: str, output_dir="features"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}.feather")
    df.reset_index().to_feather(path)
    print(f"[✓] Saved {symbol} to {path}")

def main():
    conn = psycopg2.connect(
        dbname="alpha_db",
        user="shiva",
        password="password",  # Change to your actual password
        host="localhost",
        port=5432
    )

    symbols = ["AAPL", "MSFT", "GOOGL"]  # Or fetch from DB if needed

    for symbol in symbols:
        print(f"[→] Processing {symbol}")
        df = fetch_ohlcv(conn, symbol)
        if len(df) < 100:
            print(f"[!] Skipping {symbol}: Not enough data")
            continue
        df_feat = compute_features(df)
        df_combined = df.join(df_feat)
        print("COLS: ",df_combined.columns)
        save_arrow(df_combined, symbol)

    conn.close()

if __name__ == "__main__":
    main()