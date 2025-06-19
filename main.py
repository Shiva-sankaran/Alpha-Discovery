import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
import sys
sys.path.append('./strategy_cpp/build')
import strategy_cpp
from rust_factors import compute_all_factors
warnings.filterwarnings('ignore')
from sdk.factor_dataset import FactorDataset

""" This code was commented and cleaned by claude.ai"""

# Install yfinance if not already installed
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call(["pip", "install", "yfinance"])
    import yfinance as yf

class AlphaFactorAnalysis:
    """
    Fixed Alpha Factor Analysis with proper Sharpe ratio calculation
    """
    
    def __init__(self, lookback_window=20, forward_return_period=5):
        self.lookback_window = lookback_window
        self.forward_return_period = forward_return_period
        self.scaler = StandardScaler()

    def generate_technical_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Use Rust backend to compute technical factors
        """
        required_cols = {"close", "volume"}
        if not required_cols.issubset(set(data.columns)):
            raise ValueError(f"Missing required columns: {required_cols - set(data.columns)}")

        try:
            # Ensure correct dtypes for Rust FFI
            close = data["close"].values.astype(np.float64)
            volume = data["volume"].values.astype(np.float64)

            # Call Rust function
            factors_dict = compute_all_factors(close, volume)

            # Convert result into DataFrame
            factors_df = pd.DataFrame(factors_dict, index=data.index)

            # Combine with original DataFrame (non-destructive)
            return pd.concat([data, factors_df], axis=1)

        except Exception as e:
            print(f"[Rust Factor Generation Error] {e}")
            return data
        
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend_strength(self, prices, period):
        """Calculate trend strength using linear regression slope"""
        def slope(x):
            if len(x) < 2:
                return 0
            return stats.linregress(range(len(x)), x)[0]
        
        return prices.rolling(period).apply(slope, raw=False)
    
    def calculate_forward_returns(self, data, periods=[1, 5, 10, 20]):
        """Calculate forward returns for different periods"""
        df = data.copy()
        
        for period in periods:
            df[f'forward_return_{period}'] = df['close'].shift(-period) / df['close'] - 1
            
        return df
    
    def calculate_strategy_returns_fixed(self, predictions, actual_returns, target_annual_vol=0.10, cost_per_unit=0.005):
        """
        Wrapper for C++ strategy return engine using pybind11 module
        """
        try:
            returns, leverage, turnover = strategy_cpp.compute_returns_cpp(
                list(predictions), list(actual_returns), target_annual_vol, cost_per_unit
            )

            return {
                'market_neutral': np.array(returns),
                'gross_returns': np.array(returns),  # in this version we don't split gross vs net
                'turnover': np.array(turnover),
                'leverage': leverage,
                # You can leave other strategies (quintile, percentile) as Python fallback
                'quintile_long_short': [0.0] * len(actual_returns),
                'percentile_long_short': [0.0] * len(actual_returns)
            }
        except Exception as e:
            print(f"[C++ Strategy Error] {e}")
            return {
                'market_neutral': [0.0] * len(actual_returns),
                'gross_returns': [0.0] * len(actual_returns),
                'turnover': [0.0] * len(actual_returns),
                'leverage': 1.0,
                'quintile_long_short': [0.0] * len(actual_returns),
                'percentile_long_short': [0.0] * len(actual_returns)
            }
   
    
    def calculate_proper_sharpe(self,strategy_returns):
        """
        Calculate Sharpe ratio properly
        """
        if len(strategy_returns) == 0 or np.std(strategy_returns) <= 1e-10:
            return 0.0
        
        # Remove any positions with zero returns for active-only Sharpe
        # strategy_returns = np.array(strategy_returns)
        # print("strategy_returns: ",strategy_returns)

        active_returns = strategy_returns[strategy_returns != 0]
        # print("active_returns: ",active_returns)
        if len(active_returns) == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        mean_return = np.mean(active_returns)
        std_return = np.std(active_returns)
        
        if std_return <= 1e-10:
            return 0.0
        
        sharpe = (mean_return / std_return) * np.sqrt(252)  # Assuming daily returns
        
        return sharpe


    def linear_regression_analysis(self, factor_data, target_return, factor_name, n_splits=5):
        """
        Fixed linear regression analysis with proper strategy implementation
        """
        # Align and clean data
        if isinstance(factor_data, pd.DataFrame):
            combined_data = pd.concat([factor_data, target_return], axis=1).dropna()
        else:
            combined_data = pd.concat([factor_data, target_return], axis=1).dropna()
        
        if len(combined_data) < 100:  # Minimum data requirement
            return None
            
        # Separate features and target
        if isinstance(factor_data, pd.DataFrame):
            X = combined_data.iloc[:, :-1].values  # All columns except last (target)
        else:
            X = combined_data.iloc[:, 0].values.reshape(-1, 1)  # First column as feature
        y = combined_data.iloc[:, -1].values  # Last column as target
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = []
        all_predictions = []
        all_actual = []
        all_strategy_returns = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            strategy_returns_dict = self.calculate_strategy_returns_fixed(y_pred, y_test)
            # print("strategy_returns_dict['market_neutral']:", strategy_returns_dict['market_neutral'])
            strategy_returns = strategy_returns_dict['market_neutral']
            avg_turnover = np.mean(strategy_returns_dict['turnover'])
            avg_leverage = strategy_returns_dict['leverage']

            # Store results
            all_predictions.extend(y_pred)
            all_actual.extend(y_test)
            all_strategy_returns.extend(strategy_returns)
            
            cv_results.append({
                'mse': mean_squared_error(y_test, y_pred),
                'coefficient': model.coef_ if len(model.coef_) == 1 else model.coef_,
                'intercept': model.intercept_
            })
        
        # print("all_strategy_returns: ",all_strategy_returns)
        # Convert to numpy arrays
        all_strategy_returns = np.array(all_strategy_returns)
        all_predictions = np.array(all_predictions)
        all_actual = np.array(all_actual)
        
        # FIXED: Proper Sharpe ratio calculation
        # Remove zero returns (neutral positions) for Sharpe calculation
        active_returns = all_strategy_returns[all_strategy_returns != 0]
        # print("active_returns", active_returns)
        
        
        if len(active_returns) > 10 and np.std(active_returns) > 1e-10:
            # Annualized Sharpe ratio (assuming daily returns)
            # sharpe_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
            sharpe_ratio = self.calculate_proper_sharpe(strategy_returns_dict['market_neutral'])  # or chosen strategy
            penalized_sharpe = sharpe_ratio / (1 + avg_turnover + avg_leverage)


        else:
            sharpe_ratio = 0
        
        # Calculate information coefficient (correlation between predictions and actual)
        
        # Additional metrics
        
        return {
            'factor_name': factor_name,
            'sharpe_ratio': self.calculate_proper_sharpe(strategy_returns),
            'ic': np.corrcoef(all_predictions, all_actual)[0, 1] if len(all_predictions) > 1 else 0,
            'hit_rate': np.mean((all_predictions > 0) == (all_actual > 0)),
            'avg_return': np.mean(strategy_returns),
            'volatility': np.std(strategy_returns),
            'n_active_positions': np.sum(strategy_returns != 0),
            'total_observations': len(all_actual),
            'avg_mse': np.mean([fold['mse'] for fold in cv_results]),
            'cv_folds': len(cv_results),
            'avg_turnover': avg_turnover,
            'avg_leverage': avg_leverage,
            'penalized_sharpe': penalized_sharpe


        }

    
    def analyze_factor_combinations(self, data_dict, target_period=5, n_splits=5, 
                                  max_combination_size=3, top_singles_for_combinations=10):
        """
        Analyze factor combinations for all stocks with fixed methodology
        """
        all_results = []
        
        print(f"Analyzing factor combinations for {len(data_dict)} stocks...")
        
        for symbol, data in data_dict.items():
            print(f"\nProcessing {symbol}...")
            
            
            # Generate factors
            factors_df = data
            # factors_df = self.generate_technical_factors(data)
            # print("factors_df.columns", factors_df.columns)
            factors_df = self.calculate_forward_returns(factors_df, periods=[target_period])
            
            # Get factor columns
            factor_columns = [col for col in factors_df.columns if col not in 
                            ['open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns', 
                                f'forward_return_{target_period}', 'dividends', 'stock splits']]
            
            target_col = f'forward_return_{target_period}'
            
            # Clean data - remove rows with any NaN in factors or target
            clean_data = factors_df[factor_columns + [target_col]].dropna()
            # print("clean_data.columns", clean_data.columns)
            if len(clean_data) < 200:  # Need sufficient data
                print(f"  Insufficient clean data for {symbol}: {len(clean_data)} rows")
                continue
            
            print(f"  Using {len(clean_data)} clean observations for {symbol}")
            
            # Test single factors first
            print(f"  Testing {len(factor_columns)} single factors...")
            single_results = []
            
            for factor_name in factor_columns:
                if factor_name in clean_data.columns:
                    factor_series = clean_data[factor_name]
                    target_series = clean_data[target_col]
                    # print("factor_name",factor_name)
                    # print("factor_series",factor_series)
                    # Skip if factor has no variation
                    if factor_series.std() < 1e-10:
                        continue
                        
                    result = self.linear_regression_analysis(
                        factor_series, 
                        target_series,
                        [factor_name],
                        n_splits=n_splits
                    )
                    
                    if result is not None:
                        result['symbol'] = symbol
                        single_results.append(result)
                        all_results.append(result)
            
            # Test combinations if we have good single factors
            if len(single_results) > 1 and max_combination_size > 1:
                # Get top single factors for combinations
                single_df = pd.DataFrame(single_results)
                # Filter for reasonable Sharpe ratios and good IC
                good_singles = single_df[
                    # (single_df['sharpe_ratio'].abs() < 10.0) &  # Reasonable Sharpe
                    # (single_df['ic'].abs() > 0.05) &           # Meaningful IC
                    (single_df['n_active_positions'] > 20)     # Sufficient trades
                ]
                
                if len(good_singles) >= 2:
                    top_factors = good_singles.nlargest(
                        min(top_singles_for_combinations, len(good_singles)), 
                        'sharpe_ratio'
                    )['factor_name'].tolist()
                    top_factors = [f[0] for f in top_factors]  # Extract factor names
                    
                    print(f"  Testing pairs from top {len(top_factors)} factors...")
                    
                    # Test pairs
                    from itertools import combinations
                    for factor1, factor2 in combinations(top_factors, 2):
                        factor_combo = [factor1, factor2]
                        factor_data = clean_data[factor_combo]
                        target_data = clean_data[target_col]
                        
                        result = self.linear_regression_analysis(
                            factor_data, 
                            target_data,
                            factor_combo,
                            n_splits=n_splits
                        )
                        
                        if result is not None:
                            result['symbol'] = symbol
                            all_results.append(result)
                
        
        # Convert to DataFrame and clean results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Add number of factors
            results_df['n_factors'] = results_df['factor_name'].apply(len)
            
            # Filter out unrealistic results
            # results_df = results_df[
            #     (results_df['sharpe_ratio'].abs() < 10.0) &  # Reasonable Sharpe ratios
            #     (results_df['ic'].abs() < 1.0) &             # Valid IC range
            #     (results_df['n_active_positions'] > 10)      # Minimum trading activity
            # ]
            
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)
            
            print(f"\nAnalysis complete! Found {len(results_df)} valid factor combinations")
            return results_df
        else:
            print("No valid results generated.")
            return pd.DataFrame()

def fetch_market_data(symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], 
                     start_date='2020-01-01', end_date=None):
    """Fetch real market data from Yahoo Finance"""
    if end_date is None:
        from datetime import datetime
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching market data for {len(symbols)} symbols from {start_date} to {end_date}...")
    
    data_dict = {}
    
    for symbol in symbols:
        try:
            print(f"  Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if len(hist_data) < 200:  # Increased minimum requirement
                print(f"  Warning: {symbol} has insufficient data ({len(hist_data)} days)")
                continue
            
            # Rename columns to match our pipeline format
            hist_data.columns = [col.lower() for col in hist_data.columns]
            hist_data.index.name = 'date'
            hist_data = hist_data.dropna()
            
            data_dict[symbol] = hist_data
            print(f"  Successfully loaded {len(hist_data)} days of data for {symbol}")
            
        except Exception as e:
            print(f"  Error fetching {symbol}: {str(e)}")
    
    print(f"\nSuccessfully loaded data for {len(data_dict)} symbols")
    return data_dict

def get_sp500_symbols(top_n=5):
    """Get top N popular stock symbols for analysis"""
    popular_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
        'UNH', 'JNJ', 'V', 'WMT', 'MA', 'PG', 'HD', 'CVX'
    ]
    return popular_stocks[:top_n]

# Main execution
if __name__ == "__main__":
    print("FIXED Alpha Factor Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AlphaFactorAnalysis(lookback_window=20, forward_return_period=5)
    
    # Get stock symbols
    symbols = get_sp500_symbols(top_n=3)
    
    # Fetch market data
    # market_data = fetch_market_data(symbols, start_date='2020-01-01')
    market_data = {}
    for symbol in symbols:
        try:
            ds = FactorDataset(symbol)
            # df = ds.get_full()
            df = ds.get_factors()
            # print("COLS: ", df.columns)
            market_data[symbol] = df
            print(f"Loaded {symbol} from feather with {len(df)} rows")
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
    
    if len(market_data) == 0:
        print("No market data was successfully fetched.")
        exit(1)
    
    # Analyze factor combinations
    print("\nStarting FIXED factor combination analysis...")
    
    results_df = analyzer.analyze_factor_combinations(
        market_data, 
        target_period=5, 
        n_splits=5,
        max_combination_size=2,  # Start with just singles and pairs
        top_singles_for_combinations=5
    )
    
    if len(results_df) > 0:
        print(f"\nFixed Analysis Complete! Found {len(results_df)} realistic factor combinations")

        # Compute only Penalized Sharpe
        if 'penalized_sharpe' not in results_df.columns:
            results_df['penalized_sharpe'] = results_df.apply(
                lambda row: row['sharpe_ratio'] / (1 + row['avg_turnover'] + row['avg_leverage']),
                axis=1
            )

        # Display results with only penalized Sharpe
        print("\n" + "="*100)
        print("TOP REALISTIC RESULTS (Penalized Sharpe Only)")
        print("="*100)
        
        print("\nTOP 15 COMBINATIONS (All Types):")
        print("-" * 100)
        print(f"{'Symbol':<8} {'Factors':<30} {'Sharpe':<8} {'IC':<6} {'HitRate':<8} {'Trades':<7}")
        print("-" * 100)
        
        top_results = results_df.sort_values("penalized_sharpe", ascending=False).head(15)
        for _, row in top_results.iterrows():
            factors_str = str(row['factor_name'])[:28] + "..." if len(str(row['factor_name'])) > 30 else str(row['factor_name'])
            print(f"{row['symbol']:<8} {factors_str:<30} {row['penalized_sharpe']:<8.3f} {row['ic']:<6.3f} {row['hit_rate']:<8.3f} {row['n_active_positions']:<7}")

        # Summary statistics
        print(f"\nSUMMARY STATISTICS (Penalized Sharpe Only):")
        print("-" * 50)
        print(f"Total valid combinations: {len(results_df)}")
        print(f"Average Penalized Sharpe: {results_df['penalized_sharpe'].mean():.3f}")
        print(f"Best Penalized Sharpe: {results_df['penalized_sharpe'].max():.3f}")
        print(f"Average IC: {results_df['ic'].mean():.3f}")
        print(f"Average Hit Rate: {results_df['hit_rate'].mean():.3f}")

        # Performance by combination size
        if 'n_factors' in results_df.columns:
            print(f"\nPERFORMANCE BY COMBINATION SIZE:")
            print("-" * 60)
            for n_factors in sorted(results_df['n_factors'].unique()):
                subset = results_df[results_df['n_factors'] == n_factors]
                combo_type = "Singles" if n_factors == 1 else f"{n_factors}-Factor Combos"
                print(f"{combo_type:<18} | Count: {len(subset):3d} | Avg Penalized Sharpe: {subset['penalized_sharpe'].mean():6.3f} | Max Penalized: {subset['penalized_sharpe'].max():6.3f}")

        # Save results
        results_df.to_csv('fixed_alpha_factor_results.csv', index=False)
        print(f"\nResults saved to 'fixed_alpha_factor_results.csv'")

    else:
        print("No valid results generated after filtering.")
