# Automated Pairs Trading System

A comprehensive pairs trading platform featuring both automated trading via Interactive Brokers and an interactive analysis dashboard. This system identifies and executes statistical arbitrage opportunities between correlated stock pairs using mean reversion strategies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Strategy Explanation](#strategy-explanation)
- [Usage Examples](#usage-examples)
- [Configuration Guide](#configuration-guide)
- [Safety & Risk Management](#safety--risk-management)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [License](#license)

## Overview

This pairs trading system combines two powerful components:

1. **Automated Trading Bot** (`ib_pairs_trading_bot.py`): Live trading execution through Interactive Brokers
2. **Interactive Dashboard** (`streamlit_pairs_dashboard.py`): Strategy backtesting, optimization, and visualization

The system uses statistical methods to identify when two historically correlated stocks diverge from their typical relationship, then trades on the expectation that they will revert to the mean.

## Features

### Trading Bot Features
- **Live Trading**: Automated execution through Interactive Brokers API
- **Multiple Spread Methods**: Hedge ratio, log ratio, and simple ratio calculations
- **Risk Management**: Stop-loss protection and position sizing
- **Real-time Monitoring**: Continuous price tracking and signal generation
- **Paper & Live Trading**: Support for both paper and live accounts
- **Comprehensive Logging**: Detailed trade and signal logs
- **Market Hours Detection**: Automatic trading only during market hours

### Dashboard Features
- **Interactive Backtesting**: Test strategies across historical data
- **Parameter Optimization**: Automated search for optimal parameters
- **Visual Analysis**: 10+ interactive charts and metrics
- **Performance Comparison**: Strategy vs S&P 500 benchmark
- **Real-time Parameter Tuning**: Adjust settings and see results instantly
- **Risk Analytics**: Drawdown, Sharpe ratio, and win rate analysis
- **Rolling Statistics**: Correlation and z-score visualization


## Installation

### Prerequisites
- Python 3.8 or higher
- Interactive Brokers account (for live trading)
- TWS (Trader Workstation) or IB Gateway installed

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### requirements.txt
```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
plotly>=5.17.0
ib_insync>=0.9.86
pytz>=2023.3
```

### Step 2: Interactive Brokers Setup (for Trading Bot)

1. **Install TWS or IB Gateway**
   - Download from [Interactive Brokers](https://www.interactivebrokers.com/)
   - Install and log in to your account

2. **Enable API Access**
   - In TWS/Gateway: File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Add `127.0.0.1` to trusted IP addresses
   - Note your port number:
     - TWS Paper: 7497
     - TWS Live: 7496
     - Gateway Paper: 4002
     - Gateway Live: 4001

3. **Disable Auto-Logoff** (optional but recommended)
   - File → Global Configuration → Lock and Exit
   - Uncheck "Auto logoff time"

## Quick Start

### Running the Analysis Dashboard

```bash
streamlit run streamlit_pairs_dashboard.py
```

Then:
1. Enter two ticker symbols (e.g., AAPL, MSFT)
2. Click "Run Analysis"
3. Explore optimized parameters and backtest results
4. Adjust parameters using sliders to test different strategies

### Running the Trading Bot

```bash
python ib_pairs_trading_bot.py
```

Follow the interactive prompts to:
1. Enter ticker symbols
2. Set strategy parameters
3. Configure IB connection
4. Start automated trading

## Components

### 1. Interactive Dashboard (`streamlit_pairs_dashboard.py`)

**Purpose**: Research, backtest, and optimize pairs trading strategies

**Key Classes**:
- `InteractivePairsTradingDashboard`: Main dashboard controller

**Main Functions**:
- `fetch_data()`: Downloads historical price data from Yahoo Finance
- `calculate_spread()`: Computes spread between stock pairs
- `calculate_z_score()`: Calculates statistical z-score of spread
- `backtest_strategy()`: Runs strategy simulation
- `run_optimization()`: Tests multiple parameter combinations

**Workflow**:
```
User Input → Data Fetch → Optimization → Display Results → Parameter Tuning → Backtest Visualization
```

### 2. Trading Bot (`ib_pairs_trading_bot.py`)

**Purpose**: Execute pairs trades automatically via Interactive Brokers

**Key Classes**:
- `IBPairsTradingBot`: Main bot controller

**Main Methods**:
- `connect()`: Establishes IB connection
- `fetch_historical_data()`: Gets price history from IB
- `get_live_prices()`: Retrieves current market prices
- `generate_signal()`: Creates trading signals based on z-score
- `execute_signal()`: Places orders through IB
- `run()`: Main trading loop

**Workflow**:
```
Connect to IB → Fetch History → Monitor Prices → Generate Signals → Execute Trades → Log Results
```

## Strategy Explanation

### What is Pairs Trading?

Pairs trading is a market-neutral strategy that profits from the mean reversion of two correlated stocks:

1. **Identify Correlation**: Find two stocks that historically move together
2. **Calculate Spread**: Measure the difference between the stocks
3. **Detect Divergence**: When the spread moves beyond normal range (high z-score)
4. **Trade the Reversion**: 
   - When spread is high → Short the expensive stock, long the cheap one
   - When spread normalizes → Close both positions for profit

### Spread Calculation Methods

#### 1. Hedge Ratio Method (Recommended)
```python
spread = stock1 - (hedge_ratio × stock2)
```
- Uses linear regression to find optimal ratio
- Automatically adjusts for price level differences
- Most statistically sound approach

#### 2. Log Ratio Method
```python
spread = log(stock1) - log(stock2)
```
- Good for stocks with different price levels
- Handles percentage changes naturally

#### 3. Simple Ratio Method
```python
spread = stock1 / stock2
```
- Easy to interpret
- Best for similarly priced stocks

### Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **Entry Threshold** | Z-score to enter position | 1.5 - 2.5σ |
| **Exit Threshold** | Z-score to close position | 0.3 - 0.7σ |
| **Stop Loss** | Maximum z-score before forced exit | 2.5 - 4.0σ |
| **Z-Score Window** | Days for z-score calculation | 30 - 90 days |
| **Hedge Ratio Window** | Days for hedge ratio (fixed at 252) | 252 days |

### Trading Signals

**Long Spread Entry** (Z-score < -entry_threshold):
- Buy Stock 1
- Sell Stock 2
- Expectation: Spread will increase

**Short Spread Entry** (Z-score > +entry_threshold):
- Sell Stock 1
- Buy Stock 2
- Expectation: Spread will decrease

**Exit** (|Z-score| < exit_threshold):
- Close both positions
- Lock in profit/loss

**Stop Loss** (|Z-score| > stop_loss):
- Emergency exit
- Prevents excessive losses

## Usage Examples

### Example 1: Research with Dashboard

```bash
# Start dashboard
streamlit run streamlit_pairs_dashboard.py

# In the browser:
# 1. Enter: AAPL, MSFT
# 2. Click "Run Analysis"
# 3. Review optimization results
# 4. Load best parameters
# 5. Examine charts and metrics
# 6. Note optimal parameters for bot
```

### Example 2: Paper Trading

```bash
# Start bot
python ib_pairs_trading_bot.py

# Interactive setup:
First ticker: AAPL
Second ticker: MSFT
Entry threshold (σ): 2.0
Exit threshold (σ): 0.5
Stop loss (σ): 3.0
Z-score window (days): 60
Select spread method: 1 (Hedge Ratio)
Port: 7497 (TWS Paper)
Check interval: 300 seconds
Position size: 100 shares

# Bot will:
# - Connect to IB
# - Fetch historical data
# - Monitor prices every 5 minutes
# - Generate and execute signals
# - Log all activity
```

### Example 3: Live Trading (Advanced)

```bash
ONLY FOR EXPERIENCED TRADERS

python ib_pairs_trading_bot.py

# Use port 7496 (TWS Live) or 4001 (Gateway Live)
# Start with small position sizes
# Monitor logs closely
# Have a plan to manually exit if needed
```

## Configuration Guide

### Dashboard Configuration

**In Sidebar**:
- **Tickers**: Any valid stock symbols
- **Date Range**: Use default (3 years) or custom range
- **Strategy Parameters**: Adjust using sliders

**Best Practices**:
- Start with well-known, liquid stocks
- Use at least 2-3 years of data
- Run full optimization before live trading
- Compare results to S&P 500 benchmark

### Bot Configuration

**Key Settings**:

```python
# Conservative Settings (Recommended for beginners)
entry_threshold = 2.5
exit_threshold = 0.5
stop_loss = 3.5
z_window = 60
check_interval = 300  # 5 minutes
position_size = 100

# Aggressive Settings (Higher risk/reward)
entry_threshold = 1.5
exit_threshold = 0.3
stop_loss = 3.0
z_window = 45
check_interval = 180  # 3 minutes
position_size = 200
```

**Port Selection**:
- **Paper Trading**: Start here! Use 7497 (TWS) or 4002 (Gateway)
- **Live Trading**: Only after successful paper trading. Use 7496 (TWS) or 4001 (Gateway)

## Safety & Risk Management

### Built-in Protections

1. **Stop Loss**: Automatic exit when z-score exceeds threshold
2. **Market Hours Check**: Trades only during US market hours
3. **Position Limits**: Fixed position sizes (configurable)
4. **Connection Monitoring**: Alerts on IB disconnection
5. **Logging**: Complete audit trail of all actions

### Risk Considerations

**Important Warnings**:

- Pairs trading is NOT risk-free
- Correlations can break down suddenly
- Both legs of the trade can move against you
- Requires significant capital for proper diversification
- Slippage and commissions impact returns
- Historical performance ≠ future results

### Recommended Risk Limits

| Risk Factor | Recommendation |
|-------------|----------------|
| **Position Size** | Max 2-5% of portfolio per pair |
| **Correlation** | Minimum 0.7 rolling correlation |
| **Max Drawdown** | Close all if portfolio down >10% |
| **Daily Loss Limit** | Stop trading if down >2% in one day |
| **Number of Pairs** | Diversify across 5-10 pairs minimum |

## Troubleshooting

### Dashboard Issues

**Problem**: "No valid strategies found"
- **Solution**: Try different ticker pairs with stronger historical correlation
- **Check**: Ensure both tickers have sufficient historical data

**Problem**: Charts not displaying
- **Solution**: Update plotly: `pip install --upgrade plotly`
- **Check**: Browser JavaScript is enabled

**Problem**: Optimization takes too long
- **Solution**: Reduce date range or parameter search space
- **Adjust**: Decrease number of parameter combinations

### Bot Issues

**Problem**: "Failed to connect to IB"
- **Solution**: 
  1. Verify TWS/Gateway is running and logged in
  2. Check API settings are enabled
  3. Confirm correct port number
  4. Add 127.0.0.1 to trusted IPs

**Problem**: "No historical data received"
- **Solution**:
  1. Check ticker symbols are valid
  2. Verify market data subscriptions in IB
  3. Try using snapshot data instead of streaming

**Problem**: Orders not executing
- **Solution**:
  1. Check if outside market hours
  2. Verify account has sufficient buying power
  3. Review IB logs for error messages
  4. Ensure stocks are tradeable (not halted)

**Problem**: "Unable to get prices from snapshot"
- **Solution**: Normal when market is closed. Bot uses last available prices

## Best Practices

### Research Phase (Use Dashboard)

1. **Screen Many Pairs**: Test 20+ combinations
2. **Check Correlation**: Aim for >0.7 minimum
3. **Review Charts**: Look for stable relationships
4. **Optimize Parameters**: Let the system find best settings
5. **Out-of-Sample Test**: Test on recent data not used in optimization
6. **Compare Benchmarks**: Strategy should beat S&P 500

### Paper Trading Phase (Use Bot)

1. **Start Small**: Use minimum position sizes
2. **Monitor Closely**: Check logs daily
3. **Track Performance**: Record all trades manually as backup
4. **Test Edge Cases**: See how bot handles gaps, halts, etc.
5. **Run for 1-3 Months**: Build confidence before going live

### Live Trading Phase

1. **Triple Check Settings**: Verify all parameters
2. **Start One Pair**: Don't trade multiple pairs immediately
3. **Set Alerts**: Monitor via phone/email
4. **Daily Review**: Check logs and performance
5. **Have Exit Plan**: Know when to stop the bot manually
6. **Regular Reoptimization**: Markets change; reoptimize quarterly

### Maintenance

- **Daily**: Review logs and positions
- **Weekly**: Check correlation stability
- **Monthly**: Analyze performance metrics
- **Quarterly**: Reoptimize parameters
- **Annually**: Evaluate strategy viability

## Performance Metrics Explained

### Sharpe Ratio
- Measures risk-adjusted returns
- Higher is better
- Good: >1.0, Excellent: >2.0

### Maximum Drawdown
- Largest peak-to-trough decline
- Lower is better
- Acceptable: <20%, Good: <10%

### Win Rate
- Percentage of profitable trades
- Typical range: 50-65%
- Higher isn't always better (depends on win/loss size)

### Total Return
- Overall percentage gain/loss
- Compare to buy-and-hold and S&P 500
- Account for transaction costs

## Support & Resources

### Getting Help

1. **Check Logs**: Most issues are explained in log files
2. **Review Documentation**: IB API docs and ib_insync documentation
3. **Community**: Stack Overflow, Quantitative Finance forums
4. **IB Support**: For API-specific issues

### Useful Links

- [IB API Documentation](https://interactivebrokers.github.io/tws-api/)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

## Logging

The bot creates detailed logs in the `logs/` directory:

```
logs/ib_bot_AAPL_MSFT_20240120_143022.log
```

**Log Contents**:
- Connection events
- Price updates
- Signal generation
- Trade execution
- Position changes
- Errors and warnings

**Log Levels**:
- `INFO`: Normal operations
- `WARNING`: Potential issues
- `ERROR`: Problems requiring attention

## Legal Disclaimer

**IMPORTANT**: 

- This software is for educational purposes only
- Not financial advice
- No guarantee of profits
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Test thoroughly in paper trading before risking real money
- Consult a licensed financial advisor before trading
- The authors assume no liability for trading losses

## License

This project is for educational purposes. Use at your own risk.

---

## Educational Notes

### Why Pairs Trading Works

Pairs trading exploits **mean reversion** - the tendency of prices to return to average levels. When two correlated stocks diverge:

1. **Statistical Edge**: The spread has historically reverted
2. **Market Neutral**: Hedged against broad market moves
3. **Lower Risk**: Both long and short positions reduce exposure

### Why It Might Fail

1. **Correlation Breakdown**: Fundamental changes to one company
2. **Regime Change**: Market conditions shift permanently
3. **Execution Risk**: Slippage, failed orders, connection issues
4. **Black Swan Events**: Unexpected market shocks

### Improving the Strategy

**Advanced Enhancements**:
- Cointegration testing instead of correlation
- Dynamic position sizing based on volatility
- Multiple timeframe analysis
- Machine learning for parameter optimization
- Portfolio-level risk management
- Sector-specific pair selection

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Maintainer**: Roopak Dasararaju

For questions, issues, or contributions, please open an issue on the repository.
