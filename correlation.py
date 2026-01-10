import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
import seaborn as sns

class PairsTradingAnalyzer:
    """
    A class to analyze stock pairs for pairs trading strategies.
    Calculates correlation, cointegration, and generates trading signals.
    """
    
    def __init__(self, ticker1, ticker2, start_date=None, end_date=None):
        """
        Initialize the analyzer with two stock tickers.
        
        Args:
            ticker1 (str): First stock ticker symbol
            ticker2 (str): Second stock ticker symbol
            start_date (str): Start date in 'YYYY-MM-DD' format (default: 1 year ago)
            end_date (str): End date in 'YYYY-MM-DD' format (default: today)
        """
        self.ticker1 = ticker1.upper()
        self.ticker2 = ticker2.upper()
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.spread = None
        self.z_score = None
        
    def fetch_data(self):
        """Download historical price data for both stocks."""
        print(f"Fetching data for {self.ticker1} and {self.ticker2}...")
        
        # Download both stocks together to get consistent data structure
        tickers = f"{self.ticker1} {self.ticker2}"
        data_raw = yf.download(tickers, start=self.start_date, end=self.end_date, 
                              progress=False, auto_adjust=False)
        
        # Extract Adj Close prices for both stocks
        if len(data_raw.columns.levels) > 1:  # Multiple tickers
            self.data = pd.DataFrame({
                self.ticker1: data_raw['Adj Close'][self.ticker1],
                self.ticker2: data_raw['Adj Close'][self.ticker2]
            }).dropna()
        else:  # Single ticker (shouldn't happen but just in case)
            raise ValueError("Please provide two different ticker symbols")
        
        
        print(f"Downloaded {len(self.data)} trading days of data")
        return self.data
    
    def calculate_correlation(self):
        """Calculate Pearson correlation coefficient between the two stocks."""
        if self.data is None:
            self.fetch_data()
        
        correlation = self.data[self.ticker1].corr(self.data[self.ticker2])
        print(f"\nPearson Correlation: {correlation:.4f}")
        return correlation
    
    def calculate_rolling_correlation(self, window=30):
        """Calculate rolling correlation over a specified window."""
        if self.data is None:
            self.fetch_data()
        
        rolling_corr = self.data[self.ticker1].rolling(window=window).corr(
            self.data[self.ticker2]
        )
        return rolling_corr
    
    def test_cointegration(self):
        """
        Perform Engle-Granger cointegration test.
        Returns test statistic, p-value, and critical values.
        """
        if self.data is None:
            self.fetch_data()
        
        from statsmodels.tsa.stattools import coint
        
        score, pvalue, _ = coint(self.data[self.ticker1], self.data[self.ticker2])
        
        print(f"\nCointegration Test Results:")
        print(f"Test Statistic: {score:.4f}")
        print(f"P-value: {pvalue:.4f}")
        
        if pvalue < 0.05:
            print("✓ Stocks are cointegrated (good for pairs trading)")
        else:
            print("✗ Stocks are NOT cointegrated (may not be suitable for pairs trading)")
        
        return score, pvalue
    
    def calculate_spread(self, method='ratio'):
        """
        Calculate the spread between two stocks.
        
        Args:
            method (str): 'ratio' for price ratio or 'difference' for price difference
        """
        if self.data is None:
            self.fetch_data()
        
        if method == 'ratio':
            self.spread = self.data[self.ticker1] / self.data[self.ticker2]
        else:
            # Calculate hedge ratio using linear regression
            slope, intercept, _, _, _ = stats.linregress(
                self.data[self.ticker2], 
                self.data[self.ticker1]
            )
            self.spread = self.data[self.ticker1] - (slope * self.data[self.ticker2])
        
        return self.spread
    
    def calculate_z_score(self, window=30):
        """Calculate z-score of the spread for trading signals."""
        if self.spread is None:
            self.calculate_spread()
        
        spread_mean = self.spread.rolling(window=window).mean()
        spread_std = self.spread.rolling(window=window).std()
        self.z_score = (self.spread - spread_mean) / spread_std
        
        return self.z_score
    
    def generate_trading_signals(self, entry_threshold=2.0, exit_threshold=0.5):
        """
        Generate trading signals based on z-score.
        
        Args:
            entry_threshold (float): Z-score threshold to enter a trade
            exit_threshold (float): Z-score threshold to exit a trade
        """
        if self.z_score is None:
            self.calculate_z_score()
        
        signals = pd.DataFrame(index=self.data.index)
        signals['z_score'] = self.z_score
        signals['signal'] = 0
        
        # Buy signal: z-score < -entry_threshold (spread is low, buy the spread)
        # Sell signal: z-score > entry_threshold (spread is high, sell the spread)
        signals.loc[self.z_score < -entry_threshold, 'signal'] = 1
        signals.loc[self.z_score > entry_threshold, 'signal'] = -1
        signals.loc[abs(self.z_score) < exit_threshold, 'signal'] = 0
        
        return signals
    
    def plot_analysis(self):
        """Create comprehensive visualization of the pairs trading analysis."""
        if self.data is None:
            self.fetch_data()
        
        if self.spread is None:
            self.calculate_spread()
        
        if self.z_score is None:
            self.calculate_z_score()
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Plot 1: Normalized prices
        normalized_data = self.data / self.data.iloc[0] * 100
        axes[0].plot(normalized_data.index, normalized_data[self.ticker1], 
                     label=self.ticker1, linewidth=2)
        axes[0].plot(normalized_data.index, normalized_data[self.ticker2], 
                     label=self.ticker2, linewidth=2)
        axes[0].set_title('Normalized Stock Prices (Base = 100)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Normalized Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Price spread
        axes[1].plot(self.spread.index, self.spread, color='purple', linewidth=1.5)
        axes[1].axhline(y=self.spread.mean(), color='r', linestyle='--', 
                        label=f'Mean: {self.spread.mean():.2f}')
        axes[1].fill_between(self.spread.index, 
                            self.spread.mean() - self.spread.std(),
                            self.spread.mean() + self.spread.std(),
                            alpha=0.2, color='gray', label='±1 Std Dev')
        axes[1].set_title('Price Spread', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Spread')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Z-score
        axes[2].plot(self.z_score.index, self.z_score, color='green', linewidth=1.5)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[2].axhline(y=2, color='r', linestyle='--', label='Entry Threshold (+2σ)')
        axes[2].axhline(y=-2, color='r', linestyle='--', label='Entry Threshold (-2σ)')
        axes[2].axhline(y=0.5, color='orange', linestyle=':', label='Exit Threshold')
        axes[2].axhline(y=-0.5, color='orange', linestyle=':')
        axes[2].fill_between(self.z_score.index, -2, 2, alpha=0.1, color='yellow')
        axes[2].set_title('Z-Score of Spread', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Z-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Rolling correlation
        rolling_corr = self.calculate_rolling_correlation(window=30)
        axes[3].plot(rolling_corr.index, rolling_corr, color='blue', linewidth=1.5)
        axes[3].axhline(y=rolling_corr.mean(), color='r', linestyle='--', 
                        label=f'Mean: {rolling_corr.mean():.3f}')
        axes[3].set_title('30-Day Rolling Correlation', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Correlation')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        if self.data is None:
            self.fetch_data()
        
        print("\n" + "="*60)
        print(f"PAIRS TRADING ANALYSIS REPORT")
        print(f"{self.ticker1} vs {self.ticker2}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print("="*60)
        
        # Correlation analysis
        correlation = self.calculate_correlation()
        rolling_corr = self.calculate_rolling_correlation(window=30)
        print(f"\n30-Day Rolling Correlation (Mean): {rolling_corr.mean():.4f}")
        print(f"30-Day Rolling Correlation (Std): {rolling_corr.std():.4f}")
        
        # Cointegration test
        self.test_cointegration()
        
        # Spread statistics
        self.calculate_spread()
        print(f"\nSpread Statistics:")
        print(f"Mean: {self.spread.mean():.4f}")
        print(f"Std Dev: {self.spread.std():.4f}")
        print(f"Min: {self.spread.min():.4f}")
        print(f"Max: {self.spread.max():.4f}")
        
        # Z-score analysis
        self.calculate_z_score()
        print(f"\nZ-Score Statistics:")
        print(f"Current Z-Score: {self.z_score.iloc[-1]:.4f}")
        print(f"Mean: {self.z_score.mean():.4f}")
        print(f"Std Dev: {self.z_score.std():.4f}")
        
        # Trading signals
        signals = self.generate_trading_signals()
        num_signals = (signals['signal'] != 0).sum()
        print(f"\nTrading Signals (±2σ threshold):")
        print(f"Total signals generated: {num_signals}")
        print(f"Buy signals: {(signals['signal'] == 1).sum()}")
        print(f"Sell signals: {(signals['signal'] == -1).sum()}")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    # Example: Analyze Coca-Cola (KO) and PepsiCo (PEP)
    analyzer = PairsTradingAnalyzer('KO', 'PEP', start_date='2023-01-01')
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    # Create visualizations
    analyzer.plot_analysis()
    
    # Access specific metrics
    print(f"\nCurrent spread value: {analyzer.spread.iloc[-1]:.4f}")
    print(f"Current z-score: {analyzer.z_score.iloc[-1]:.4f}")