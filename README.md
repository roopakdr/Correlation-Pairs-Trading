Correlation-Pairs-Trading

A Python implementation of a correlation-based pairs trading system, including pair selection, visualization, and a live trading bot using the Interactive Brokers (IBKR) API.
​

Overview

This project focuses on identifying and trading highly correlated equity pairs using a mean-reversion style strategy.
​
The repository currently includes the core correlation analytics plus an IBKR-ready algorithmic trading script.
​

Features

Correlation analysis for a universe of symbols to identify candidate pairs.
​

Price ratio normalization and visualization utilities to inspect spread behavior over time.
​

Interactive Brokers bot implementation (algo.py) for live or paper trading of selected pairs.
​

Modular structure so you can plug in your own universe, signal logic, and risk management rules.
​

Project Structure

correlation.py:

Computes rolling correlations and related statistics for a set of tickers.

Produces charts for price levels, ratios, and correlation dynamics to assist in pair selection.
​

algo.py:

Implements the IBKR pairs trading bot using signals derived from the correlation / spread behavior.

Handles order submission and position management through the Interactive Brokers API.

​

Requirements
Python 3.9+ (recommended).
​

Typical scientific Python stack:

pandas, numpy, matplotlib / plotly for analysis and charts.

ib_insync (or IB API wrapper of your choice) for Interactive Brokers connectivity.
​

Install dependencies (example):

bash
pip install pandas numpy matplotlib plotly ib-insync
Adjust package names to match the imports in algo.py and correlation.py.
​

Usage
1. Run correlation analysis
Use correlation.py to explore a universe of tickers and shortlist promising pairs based on correlation and spread behavior.

bash
python correlation.py
Typical workflow:

Define your ticker universe and lookback parameters inside correlation.py.

Run the script to generate correlation metrics and plots.

Inspect charts and statistics to select 1–3 candidate pairs to trade.
​

2. Configure IBKR connection
Before running the bot:

Ensure IB Gateway or TWS is running and API access is enabled.

Update host, port, and client ID parameters in algo.py.

Specify the chosen pair tickers, trade size, and any risk limits (e.g., max open positions, stop thresholds).
​

3. Run the pairs trading bot
bash
python algo.py
The bot will:

Monitor the selected pair in real time.

Compute signal metrics (e.g., price ratio / spread relative to its recent mean).

Open and close pair trades when signals breach configured thresholds.
​

Start in paper trading mode on IBKR and verify behavior before deploying with real capital.
​

Configuration
Key parameters you may want to customize:

Universe / tickers used for correlation analysis.
​

Lookback windows for correlation and spread statistics.
​

Entry / exit thresholds for the trading signals in algo.py.
​

Position sizing and leverage constraints aligned with your risk preferences.
​

These are typically defined as constants or configuration sections near the top of each script.

Notes and Disclaimer
This codebase is for educational and research purposes only and is not investment advice.
​

Trading involves substantial risk, including the risk of loss of capital. Use at your own risk and always test thoroughly in a simulated environment first.
