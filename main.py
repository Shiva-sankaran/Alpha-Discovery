import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


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
        
    def generate_technical_factors(self, data):
        """
        Generate technical analysis factors from price and volume data
        """
        df = data.copy()
        
        # Price-based factors
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum factors
        for period in [5, 10, 20, 60]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # Volatility factors
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'realized_vol_{period}'] = np.sqrt(252) * df['returns'].rolling(period).std()
        
        # Volume factors
        df['volume_ma_ratio_10'] = df['volume'] / df['volume'].rolling(10).mean()
        df['volume_ma_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        
        # Microstructure factors
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Mean reversion factors
        for period in [10, 20, 50]:
            df[f'mean_reversion_{period}'] = (df['close'] - df['close'].rolling(period).mean()) / df['close'].rolling(period).std()
        
        # Trend strength factors
        for period in [10, 20]:
            df[f'trend_strength_{period}'] = self._calculate_trend_strength(df['close'], period)
        
        return df
    
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
        Proper implementation of factor-based strategy returns with volatility targeting and transaction cost penalty
        """
        predictions = np.asarray(predictions)
        actual_returns = np.asarray(actual_returns)

        # Method 1: Market Neutral (Volatility Targeted)
        ranks = stats.rankdata(predictions)
        n = len(ranks)
        weights = (ranks - (n + 1) / 2) / (n / 2)  # Normalize to [-1, 1]

        gross_returns = weights * actual_returns

        # Compute daily volatility and scale to target annualized vol
        daily_vol = np.std(gross_returns)
        if daily_vol > 1e-8:
            target_daily_vol = target_annual_vol / np.sqrt(252)
            leverage = target_daily_vol / daily_vol
        else:
            leverage = 1.0

        weights *= leverage
        gross_returns = weights * actual_returns

        # Compute turnover cost
        turnover_cost = np.zeros_like(weights)
        turnover_cost[1:] = cost_per_unit * np.abs(weights[1:] - weights[:-1])
        net_returns = gross_returns - turnover_cost

        # Method 2: Quintile Long-Short (unchanged)
        try:
            quintiles = pd.qcut(predictions, q=5, labels=False, duplicates='drop')
            quintile_returns = np.zeros_like(actual_returns)
            if len(np.unique(quintiles)) >= 2:
                top_quintile = quintiles == np.max(quintiles)
                bottom_quintile = quintiles == np.min(quintiles)
                quintile_returns[top_quintile] = actual_returns[top_quintile]
                quintile_returns[bottom_quintile] = -actual_returns[bottom_quintile]
        except:
            quintile_returns = np.zeros_like(actual_returns)

        # Method 3: Percentile-based Long-Short (unchanged)
        p_high = np.percentile(predictions, 80)
        p_low = np.percentile(predictions, 20)
        percentile_returns = np.zeros_like(actual_returns)
        long_mask = predictions >= p_high
        short_mask = predictions <= p_low
        percentile_returns[long_mask] = actual_returns[long_mask]
        percentile_returns[short_mask] = -actual_returns[short_mask]

        return {
            'market_neutral': net_returns,
            'gross_returns': gross_returns,
            'turnover': turnover_cost,
            'leverage': leverage,
            'quintile_long_short': quintile_returns,
            'percentile_long_short': percentile_returns
        }

    
    def calculate_proper_sharpe(self,strategy_returns):
        """
        Calculate Sharpe ratio properly
        """
        if len(strategy_returns) == 0 or np.std(strategy_returns) <= 1e-10:
            return 0.0
        
        # Remove any positions with zero returns for active-only Sharpe
        active_returns = strategy_returns[strategy_returns != 0]
        
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
        
        # Convert to numpy arrays
        all_strategy_returns = np.array(all_strategy_returns)
        all_predictions = np.array(all_predictions)
        all_actual = np.array(all_actual)
        
        # FIXED: Proper Sharpe ratio calculation
        # Remove zero returns (neutral positions) for Sharpe calculation
        active_returns = all_strategy_returns[all_strategy_returns != 0]

        
        
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
            
            try:
                # Generate factors
                factors_df = self.generate_technical_factors(data)
                factors_df = self.calculate_forward_returns(factors_df, periods=[target_period])
                
                # Get factor columns
                factor_columns = [col for col in factors_df.columns if col not in 
                                ['open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns', 
                                 f'forward_return_{target_period}', 'dividends', 'stock splits']]
                
                target_col = f'forward_return_{target_period}'
                
                # Clean data - remove rows with any NaN in factors or target
                clean_data = factors_df[factor_columns + [target_col]].dropna()
                
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
                        (single_df['sharpe_ratio'].abs() < 5.0) &  # Reasonable Sharpe
                        (single_df['ic'].abs() > 0.05) &           # Meaningful IC
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
                
            except Exception as e:
                print(f"  Error processing {symbol}: {str(e)}")
                continue
        
        # Convert to DataFrame and clean results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Add number of factors
            results_df['n_factors'] = results_df['factor_name'].apply(len)
            
            # Filter out unrealistic results
            results_df = results_df[
                (results_df['sharpe_ratio'].abs() < 10.0) &  # Reasonable Sharpe ratios
                (results_df['ic'].abs() < 1.0) &             # Valid IC range
                (results_df['n_active_positions'] > 10)      # Minimum trading activity
            ]
            
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
    market_data = fetch_market_data(symbols, start_date='2020-01-01')
    
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
