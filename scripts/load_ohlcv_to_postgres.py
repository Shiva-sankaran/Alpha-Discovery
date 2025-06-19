import yfinance as yf
import psycopg2
import pandas as pd

def fetch_data(symbol, start="2010-01-01"):
    import datetime
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=4000)
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

    df = df.reset_index()
    df["symbol"] = symbol
    df = df[["symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
    return df

def insert_to_postgres(df, conn):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute(
                """
                INSERT INTO ohlcv (symbol, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING;
                """,
                tuple(row)
            )
        except Exception as e:
            print(f"Insert error: {e}")
    conn.commit()
    cursor.close()

def main():
    conn = psycopg2.connect(
        dbname="alpha_db",
        user="shiva",
        password="password",  # ← update with your actual password
        host="localhost",
        port=5432
    )

    symbols = ["AAPL", "MSFT", "GOOGL"]  # Add more as needed
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        df = fetch_data(symbol)
        print(f"Inserting {len(df)} rows...")
        insert_to_postgres(df, conn)
        print(f"{symbol} done.")

    conn.close()

if __name__ == "__main__":
    main()