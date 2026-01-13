import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class InteractivePairsTradingDashboard:
    """
    Interactive Streamlit dashboard for pairs trading analysis.
    """
    
    def __init__(self, ticker1, ticker2, start_date=None, end_date=None):
        self.ticker1 = ticker1.upper()
        self.ticker2 = ticker2.upper()
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
            
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.sp500_data = None
        self.optimization_results = None
        
    def fetch_data(self):
        """Download historical price data for both stocks."""
        with st.spinner(f'📊 Fetching data for {self.ticker1} and {self.ticker2}...'):
            tickers = f"{self.ticker1} {self.ticker2}"
            data_raw = yf.download(tickers, start=self.start_date, end=self.end_date, 
                                  progress=False, auto_adjust=False)
            
            if len(data_raw.columns.levels) > 1:
                self.data = pd.DataFrame({
                    self.ticker1: data_raw['Adj Close'][self.ticker1],
                    self.ticker2: data_raw['Adj Close'][self.ticker2]
                }).dropna()
            else:
                st.error("Please provide two different ticker symbols")
                return None
            
            st.success(f"✅ Downloaded {len(self.data)} trading days")
            return self.data
        
    def fetch_sp500(self):
        """Download S&P 500 data for benchmark comparison."""
        sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, 
                           progress=False, auto_adjust=False)
        self.sp500_data = sp500['Adj Close']
        return self.sp500_data
        
    def calculate_spread(self, method='hedge_ratio', lookback=252):
        """Calculate spread between stocks."""
        if method == 'ratio':
            return self.data[self.ticker1] / self.data[self.ticker2], None
        elif method == 'log_ratio':
            return np.log(self.data[self.ticker1]) - np.log(self.data[self.ticker2]), None
        elif method == 'hedge_ratio':
            hedge_ratios = []
            for i in range(lookback, len(self.data)):
                window_data = self.data.iloc[i-lookback:i]
                slope, _, _, _, _ = stats.linregress(
                    window_data[self.ticker2], window_data[self.ticker1]
                )
                hedge_ratios.append(slope)
            
            hedge_ratio = pd.Series([np.nan]*lookback + hedge_ratios, index=self.data.index)
            return self.data[self.ticker1] - (hedge_ratio * self.data[self.ticker2]), hedge_ratio
        else:
            return self.data[self.ticker1] - self.data[self.ticker2], None
            
    def calculate_z_score(self, spread, window=60):
        """Calculate z-score of spread."""
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        return (spread - spread_mean) / spread_std
    
    def backtest_strategy(self, entry_threshold, exit_threshold, stop_loss, 
                         spread_method, z_window):
        """Backtest with given parameters."""
        spread, hedge_ratio = self.calculate_spread(method=spread_method, lookback=z_window)
        if hedge_ratio is None:
            hedge_ratio = 1.0
        
        z_score = self.calculate_z_score(spread, window=z_window)
        
        signals = pd.DataFrame(index=self.data.index)
        signals['z_score'] = z_score
        signals['position'] = 0
        signals['trade_id'] = 0
        
        current_position = 0
        trade_id = 0
        trade_entry_idx = None
        
        for i in range(1, len(signals)):
            if pd.isna(signals['z_score'].iloc[i]):
                signals.iloc[i, signals.columns.get_loc('position')] = current_position
                signals.iloc[i, signals.columns.get_loc('trade_id')] = trade_id
                continue
            
            z = signals['z_score'].iloc[i]
            prev_position = current_position
            
            if current_position != 0 and abs(z) > stop_loss:
                current_position = 0
                trade_id += 1
                trade_entry_idx = None
            elif current_position == 0:
                if z < -entry_threshold:
                    current_position = 1
                    if prev_position == 0:
                        trade_id += 1
                        trade_entry_idx = i
                elif z > entry_threshold:
                    current_position = -1
                    if prev_position == 0:
                        trade_id += 1
                        trade_entry_idx = i
            elif abs(z) < exit_threshold:
                current_position = 0
                trade_id += 1
                trade_entry_idx = None
            
            signals.iloc[i, signals.columns.get_loc('position')] = current_position
            signals.iloc[i, signals.columns.get_loc('trade_id')] = trade_id
        
        returns1 = self.data[self.ticker1].pct_change()
        returns2 = self.data[self.ticker2].pct_change()
        
        if isinstance(hedge_ratio, pd.Series):
            strategy_returns = signals['position'] * (returns1 - hedge_ratio * returns2)
        else:
            strategy_returns = signals['position'] * (returns1 - hedge_ratio * returns2)
        
        strategy_returns = strategy_returns.fillna(0)
        strategy_cumulative = (1 + strategy_returns).cumprod()
        
        sp500_returns = self.sp500_data.pct_change().fillna(0)
        common_index = strategy_returns.index.intersection(sp500_returns.index)
        
        strategy_returns_aligned = strategy_returns.loc[common_index]
        sp500_returns_aligned = sp500_returns.loc[common_index]
        
        strategy_cumulative = (1 + strategy_returns_aligned).cumprod()
        sp500_cumulative = (1 + sp500_returns_aligned).cumprod()
        
        total_return = (float(strategy_cumulative.iloc[-1]) - 1) * 100 if len(strategy_cumulative) > 0 else 0
        sp500_return = (float(sp500_cumulative.iloc[-1]) - 1) * 100 if len(sp500_cumulative) > 0 else 0
        sharpe = (strategy_returns_aligned.mean() / strategy_returns_aligned.std()) * np.sqrt(252) if strategy_returns_aligned.std() > 0 else 0
        
        cummax = strategy_cumulative.cummax()
        drawdown = (strategy_cumulative - cummax) / cummax * 100
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0
        
        signals_aligned = signals.loc[common_index]
        trade_returns = []
        
        for tid in range(1, signals_aligned['trade_id'].max() + 1):
            trade_mask = signals_aligned['trade_id'] == tid
            trade_return = strategy_returns_aligned[trade_mask].sum()
            if abs(trade_return) > 1e-10:
                trade_returns.append(trade_return)
        
        num_trades = len(trade_returns)
        win_rate = (sum(1 for r in trade_returns if r > 0) / num_trades * 100) if num_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sp500_return': sp500_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'strategy_cumulative': strategy_cumulative,
            'sp500_cumulative': sp500_cumulative,
            'z_score': z_score,
            'spread': spread,
            'signals': signals,
            'drawdown': drawdown
        }
    
    def run_optimization(self):
        """Run full parameter optimization."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        entry_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        exit_thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]
        z_windows = [20, 30, 45, 60, 90, 120]
        spread_methods = ['hedge_ratio', 'log_ratio', 'ratio']
        stop_losses = [2.5, 3.0, 3.5, 4.0]
        
        results = []
        total = len(spread_methods) * len(z_windows) * len(entry_thresholds) * len(exit_thresholds) * len(stop_losses)
        count = 0
        
        status_text.text(f"Starting optimization of {total} parameter combinations...")
        
        for spread_method in spread_methods:
            for z_window in z_windows:
                for entry in entry_thresholds:
                    for exit in exit_thresholds:
                        if exit >= entry:
                            continue
                        for stop_loss in stop_losses:
                            if stop_loss <= entry:
                                continue
                            
                            count += 1
                            progress = count / total
                            progress_bar.progress(progress)
                            
                            if count % 50 == 0:
                                status_text.text(f"Progress: {count}/{total} ({progress*100:.1f}%) - Found {len(results)} valid strategies")
                            
                            try:
                                backtest = self.backtest_strategy(
                                    entry, exit, stop_loss, spread_method, z_window
                                )
                                
                                if backtest['num_trades'] >= 5:
                                    results.append({
                                        'spread_method': spread_method,
                                        'z_window': z_window,
                                        'entry': entry,
                                        'exit': exit,
                                        'stop_loss': stop_loss,
                                        'sharpe': backtest['sharpe'],
                                        'total_return': backtest['total_return'],
                                        'max_dd': backtest['max_dd'],
                                        'win_rate': backtest['win_rate'],
                                        'num_trades': backtest['num_trades']
                                    })
                            except Exception as e:
                                if count % 100 == 0:
                                    status_text.text(f"Error in optimization: {str(e)}")
                                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if len(results) > 0:
            st.success(f"✅ Optimization complete! Tested {len(results)} valid strategies.")
            return pd.DataFrame(results)
        else:
            st.error("❌ No valid strategies found. Try different tickers or parameters.")
            return pd.DataFrame()


def main():
    st.set_page_config(page_title="Pairs Trading Dashboard", layout="wide", page_icon="📊")
    
    st.title("🎯 Interactive Pairs Trading Dashboard")
    
    if 'spread_method' not in st.session_state:
        st.session_state.spread_method = 'hedge_ratio'
    if 'z_window' not in st.session_state:
        st.session_state.z_window = 60
    if 'entry' not in st.session_state:
        st.session_state.entry = 1.5
    if 'exit' not in st.session_state:
        st.session_state.exit = 0.5
    if 'stop_loss' not in st.session_state:
        st.session_state.stop_loss = 3.0
    
    with st.sidebar:
        st.header("📌 Configuration")
        
        ticker1 = st.text_input("First Ticker", value="AAPL").upper()
        ticker2 = st.text_input("Second Ticker", value="MSFT").upper()
        
        st.subheader("Date Range")
        use_default = st.checkbox("Use last 3 years", value=True)
        
        if not use_default:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*3))
            end_date = st.date_input("End Date", value=datetime.now())
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')
        else:
            start_date = None
            end_date = None
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if st.button("🔄 Reset Dashboard", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    should_run = run_analysis or ('dashboard' in st.session_state and st.session_state.dashboard is not None)
    
    if not should_run:
        st.info("👈 Enter ticker symbols in the sidebar and click 'Run Analysis' to begin!")
        return
    
    cache_key = f"{ticker1}_{ticker2}_{start_date}_{end_date}"
    
    if 'dashboard' not in st.session_state or st.session_state.get('cache_key') != cache_key:
        if run_analysis or 'dashboard' not in st.session_state:
            dashboard = InteractivePairsTradingDashboard(ticker1, ticker2, start_date, end_date)
            
            if dashboard.fetch_data() is None:
                return
            
            dashboard.fetch_sp500()
            
            with st.spinner("🔄 Running optimization... This may take a minute."):
                dashboard.optimization_results = dashboard.run_optimization()
            
            st.session_state.dashboard = dashboard
            st.session_state.cache_key = cache_key
            st.session_state.optimization_done = True
    
    dashboard = st.session_state.dashboard
    
    if dashboard.optimization_results is None or len(dashboard.optimization_results) == 0:
        st.error("❌ Optimization failed. Please try different ticker symbols.")
        return
    
    df = dashboard.optimization_results
    best_idx = df['sharpe'].idxmax()
    worst_idx = df['sharpe'].idxmin()
    best = df.loc[best_idx]
    worst = df.loc[worst_idx]
    
    if 'params_loaded' not in st.session_state:
        st.session_state.spread_method = best['spread_method']
        st.session_state.z_window = int(best['z_window'])
        st.session_state.entry = float(best['entry'])
        st.session_state.exit = float(best['exit'])
        st.session_state.stop_loss = float(best['stop_loss'])
        st.session_state.params_loaded = True
    
    st.header("🎛️ Strategy Parameters")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("")
        if st.button("🏆 Load Best", use_container_width=True):
            st.session_state.spread_method = best['spread_method']
            st.session_state.z_window = int(best['z_window'])
            st.session_state.entry = float(best['entry'])
            st.session_state.exit = float(best['exit'])
            st.session_state.stop_loss = float(best['stop_loss'])
            st.rerun()
        
        if st.button("⚠️ Load Worst", use_container_width=True):
            st.session_state.spread_method = worst['spread_method']
            st.session_state.z_window = int(worst['z_window'])
            st.session_state.entry = float(worst['entry'])
            st.session_state.exit = float(worst['exit'])
            st.session_state.stop_loss = float(worst['stop_loss'])
            st.rerun()
    
    with col1:
        cols = st.columns(5)
        
        with cols[0]:
            spread_method = st.selectbox(
                "Spread Method",
                options=['hedge_ratio', 'log_ratio', 'ratio'],
                index=['hedge_ratio', 'log_ratio', 'ratio'].index(st.session_state.spread_method),
                format_func=lambda x: {
                    'hedge_ratio': 'Hedge Ratio',
                    'log_ratio': 'Log Ratio',
                    'ratio': 'Simple Ratio'
                }[x],
                key='spread_select'
            )
            st.session_state.spread_method = spread_method
        
        with cols[1]:
            z_window = st.slider(
                "Z-Score Window", 
                20, 120, 
                st.session_state.z_window, 
                10,
                key='z_slider'
            )
            st.session_state.z_window = z_window
        
        with cols[2]:
            entry = st.slider(
                "Entry Threshold (σ)", 
                0.5, 3.0, 
                st.session_state.entry, 
                0.5,
                key='entry_slider'
            )
            st.session_state.entry = entry
        
        with cols[3]:
            exit = st.slider(
                "Exit Threshold (σ)", 
                0.1, 1.0, 
                st.session_state.exit, 
                0.1,
                key='exit_slider'
            )
            st.session_state.exit = exit
        
        with cols[4]:
            stop_loss = st.slider(
                "Stop Loss (σ)", 
                2.5, 4.0, 
                st.session_state.stop_loss, 
                0.5,
                key='stop_slider'
            )
            st.session_state.stop_loss = stop_loss
    
    result = dashboard.backtest_strategy(entry, exit, stop_loss, spread_method, z_window)
    
    st.header("📊 Performance Metrics")
    metric_cols = st.columns(6)
    
    sharpe_rank = (df['sharpe'] > result['sharpe']).sum() + 1
    return_rank = (df['total_return'] > result['total_return']).sum() + 1
    
    with metric_cols[0]:
        st.metric(
            "Total Return",
            f"{result['total_return']:.2f}%",
            f"Rank: {return_rank}/{len(df)}"
        )
    
    with metric_cols[1]:
        st.metric(
            "S&P 500 Return",
            f"{result['sp500_return']:.2f}%",
            f"{result['total_return'] - result['sp500_return']:.2f}% vs SPY"
        )
    
    with metric_cols[2]:
        st.metric(
            "Sharpe Ratio",
            f"{result['sharpe']:.2f}",
            f"Rank: {sharpe_rank}/{len(df)}"
        )
    
    with metric_cols[3]:
        st.metric(
            "Max Drawdown",
            f"{abs(result['max_dd']):.2f}%"
        )
    
    with metric_cols[4]:
        st.metric(
            "Win Rate",
            f"{result['win_rate']:.1f}%"
        )
    
    with metric_cols[5]:
        st.metric(
            "Trades",
            f"{result['num_trades']}"
        )
    
    st.header("📈 Analysis Charts")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Normalized Stock Prices")
        norm_data = dashboard.data / dashboard.data.iloc[0] * 100
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=norm_data.index, y=norm_data[dashboard.ticker1],
            name=dashboard.ticker1, line=dict(color='#3b82f6', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=norm_data.index, y=norm_data[dashboard.ticker2],
            name=dashboard.ticker2, line=dict(color='#ef4444', width=2)
        ))
        fig1.update_layout(
            height=400, 
            hovermode='x unified', 
            yaxis_title="Normalized Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        st.subheader("Cumulative Returns")
        
        strategy_dates = result['strategy_cumulative'].index
        strategy_returns = (result['strategy_cumulative'] - 1) * 100
        
        final_strategy = float(strategy_returns.iloc[-1]) if len(strategy_returns) > 0 else 0.0
        
        fig2 = go.Figure()
        
        # Strategy returns
        fig2.add_trace(go.Scatter(
            x=strategy_dates,
            y=strategy_returns,
            name='Pairs Strategy',
            line=dict(color='#10b981', width=2),
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add final value annotation
        if len(strategy_dates) > 0:
            fig2.add_annotation(
                x=strategy_dates[-1],
                y=final_strategy,
                text=f"{final_strategy:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=-40,
                ay=-30,
                bgcolor="#10b981",
                font=dict(color="white", size=11)
            )
        
        fig2.update_layout(
            height=400,
            hovermode='x unified',
            yaxis_title="Cumulative Return (%)",
            xaxis_title="Date",
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.subheader("Price Ratio with Trading Signals")
        
        price_ratio = dashboard.data[dashboard.ticker1] / dashboard.data[dashboard.ticker2]
        
        # Normalize the price ratio so mean = 1
        price_ratio_mean = price_ratio.mean()
        price_ratio_normalized = price_ratio / price_ratio_mean
        
        position_changes = result['signals']['position'].diff().fillna(0)
        
        # Get entry and exit points
        entry_signals = result['signals'][(position_changes != 0) & (result['signals']['position'] != 0)].index
        exit_signals = result['signals'][(position_changes != 0) & (result['signals']['position'] == 0)].index
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=price_ratio_normalized.index, 
            y=price_ratio_normalized,
            name='Price Ratio',
            line=dict(color='#3b82f6', width=2),
            hovertemplate='Date: %{x}<br>Normalized Ratio: %{y:.4f}<extra></extra>'
        ))
        
        if len(entry_signals) > 0:
            fig3.add_trace(go.Scatter(
                x=entry_signals,
                y=price_ratio_normalized.loc[entry_signals],
                mode='markers',
                name='Entry Signal',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color='#10b981',
                    line=dict(color='darkgreen', width=2)
                ),
                hovertemplate='ENTRY<br>Date: %{x}<br>Normalized Ratio: %{y:.4f}<extra></extra>'
            ))
        
        if len(exit_signals) > 0:
            fig3.add_trace(go.Scatter(
                x=exit_signals,
                y=price_ratio_normalized.loc[exit_signals],
                mode='markers',
                name='Exit Signal',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='#ef4444',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate='EXIT<br>Date: %{x}<br>Normalized Ratio: %{y:.4f}<extra></extra>'
            ))
        
        ratio_ma = price_ratio_normalized.rolling(60).mean()
        fig3.add_trace(go.Scatter(
            x=ratio_ma.index, 
            y=ratio_ma,
            name='60-day MA',
            line=dict(color='#f59e0b', width=2, dash='dash'),
            hovertemplate='MA: %{y:.4f}<extra></extra>'
        ))
        
        # Add horizontal line at y=1 to show the mean
        fig3.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.7,
                      annotation_text="Mean = 1", annotation_position="right")
        
        fig3.update_layout(
            height=400, 
            hovermode='x unified', 
            yaxis_title=f"Normalized Price Ratio ({dashboard.ticker1}/{dashboard.ticker2})",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with chart_col4:
        st.subheader("Spread Z-Score with Entry/Exit Signals")
        
        entry_z = result['z_score'].loc[entry_signals] if len(entry_signals) > 0 else pd.Series()
        exit_z = result['z_score'].loc[exit_signals] if len(exit_signals) > 0 else pd.Series()
        
        fig4 = go.Figure()
        
        # Z-score line
        fig4.add_trace(go.Scatter(
            x=result['z_score'].index, 
            y=result['z_score'],
            name='Z-Score', 
            line=dict(color='#8b5cf6', width=2),
            hovertemplate='Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
        ))
        
        # Entry signals
        if len(entry_z) > 0:
            fig4.add_trace(go.Scatter(
                x=entry_z.index,
                y=entry_z,
                mode='markers',
                name='Entry Signal',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color='#10b981',
                    line=dict(color='darkgreen', width=2)
                ),
                hovertemplate='ENTRY<br>Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
            ))
        
        # Exit signals
        if len(exit_z) > 0:
            fig4.add_trace(go.Scatter(
                x=exit_z.index,
                y=exit_z,
                mode='markers',
                name='Exit Signal',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='#ef4444',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate='EXIT<br>Date: %{x}<br>Z-Score: %{y:.2f}<extra></extra>'
            ))
        
        # Add threshold lines
        fig4.add_hline(y=entry, line_dash="dash", line_color="#10b981", opacity=0.5, 
                      annotation_text=f"+{entry}σ (Entry)", annotation_position="right")
        fig4.add_hline(y=-entry, line_dash="dash", line_color="#10b981", opacity=0.5,
                      annotation_text=f"-{entry}σ (Entry)", annotation_position="right")
        fig4.add_hline(y=exit, line_dash="dot", line_color="#ef4444", opacity=0.5,
                      annotation_text=f"+{exit}σ (Exit)", annotation_position="right")
        fig4.add_hline(y=-exit, line_dash="dot", line_color="#ef4444", opacity=0.5,
                      annotation_text=f"-{exit}σ (Exit)", annotation_position="right")
        fig4.add_hline(y=0, line_color="gray", line_width=1)
        
        fig4.update_layout(
            height=400, 
            hovermode='x unified', 
            yaxis_title="Z-Score",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    chart_col5, chart_col6 = st.columns(2)
    
    with chart_col5:
        st.subheader("Z-Score Timeline")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=result['z_score'].index, y=result['z_score'],
            name='Z-Score', line=dict(color='#10b981', width=2)
        ))
        fig5.add_hline(y=entry, line_dash="dash", line_color="orange", annotation_text=f"Entry ±{entry}σ")
        fig5.add_hline(y=-entry, line_dash="dash", line_color="orange")
        fig5.add_hline(y=exit, line_dash="dot", line_color="green", annotation_text=f"Exit ±{exit}σ")
        fig5.add_hline(y=-exit, line_dash="dot", line_color="green")
        fig5.add_hline(y=0, line_color="black", line_width=1)
        fig5.update_layout(
            height=400, 
            hovermode='x unified', 
            yaxis_title="Z-Score",
            showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with chart_col6:
        st.subheader("Rolling Correlation (60-day)")
        rolling_corr = dashboard.data[dashboard.ticker1].rolling(60).corr(dashboard.data[dashboard.ticker2])
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            name='Correlation', line=dict(color='#3b82f6', width=2)
        ))
        fig6.add_hline(y=rolling_corr.mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {rolling_corr.mean():.3f}")
        fig6.update_layout(
            height=400, 
            hovermode='x unified', 
            yaxis_title="Correlation",
            showlegend=False
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    chart_col7, chart_col8 = st.columns(2)
    
    with chart_col7:
        st.subheader("Strategy Drawdown")
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=result['drawdown'].index, y=result['drawdown'],
            fill='tozeroy', name='Drawdown',
            line=dict(color='#ef4444', width=0),
            fillcolor='rgba(239, 68, 68, 0.3)'
        ))
        fig7.update_layout(
            height=400, 
            hovermode='x unified', 
            yaxis_title="Drawdown (%)",
            showlegend=False
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    with chart_col8:
        st.subheader("Z-Score Distribution")
        z_clean = result['z_score'].dropna()
        fig8 = go.Figure()
        fig8.add_trace(go.Histogram(
            x=z_clean, nbinsx=50,
            marker_color='#06b6d4', opacity=0.7
        ))
        fig8.update_layout(
            height=400, 
            xaxis_title="Z-Score", 
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig8, use_container_width=True)
    
    st.subheader("Parameter Sensitivity Analysis")
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        param_df = df[(df['spread_method'] == spread_method) & (df['z_window'] == z_window)]
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(
            x=param_df['entry'], y=param_df['sharpe'],
            mode='markers',
            marker=dict(
                size=10,
                color=param_df['total_return'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return %")
            ),
            text=param_df['total_return'].round(2),
            hovertemplate='Entry: %{x}<br>Sharpe: %{y:.2f}<br>Return: %{text}%<extra></extra>'
        ))
        fig9.add_trace(go.Scatter(
            x=[entry], y=[result['sharpe']],
            mode='markers', name='Current',
            marker=dict(size=20, color='red', symbol='star')
        ))
        fig9.update_layout(
            title="Entry Threshold vs Sharpe Ratio",
            height=400,
            xaxis_title="Entry Threshold",
            yaxis_title="Sharpe Ratio",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    with param_col2:
        total_entries = len(entry_signals)
        
        st.markdown("#### 📊 Signal Statistics")
        st.write(f"**Total Entry Signals:** {total_entries} 🟢")
        st.write(f"**Exit Signals:** {len(exit_signals)} 🔴")
        st.write(f"**Complete Trades:** {result['num_trades']}")
        st.write(f"**Win Rate:** {result['win_rate']:.1f}%")
        
        if result['num_trades'] > 0:
            avg_trade_length = len(result['signals']) / result['num_trades']
            st.write(f"**Avg Trade Duration:** {avg_trade_length:.1f} days")
        
        st.markdown("---")
        st.markdown("**Signal Interpretation:**")
        st.write("🟢 **Entry** = Z-score crosses entry threshold (±{:.1f}σ)".format(entry))
        st.write("   • Opens pair: Long one stock, Short the other")
        st.write("🔴 **Exit** = Z-score reverts to exit threshold (±{:.1f}σ)".format(exit))
        st.write("   • Closes both positions")
    
    st.header("🏆 Optimization Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.subheader("Best Parameters")
        st.write(f"**Method:** {best['spread_method']}")
        st.write(f"**Z-Window:** {int(best['z_window'])}")
        st.write(f"**Entry/Exit/Stop:** {best['entry']}/{best['exit']}/{best['stop_loss']}")
        st.write(f"**Sharpe:** {best['sharpe']:.2f}")
        st.write(f"**Return:** {best['total_return']:.2f}%")
    
    with summary_col2:
        st.subheader("Worst Parameters")
        st.write(f"**Method:** {worst['spread_method']}")
        st.write(f"**Z-Window:** {int(worst['z_window'])}")
        st.write(f"**Entry/Exit/Stop:** {worst['entry']}/{worst['exit']}/{worst['stop_loss']}")
        st.write(f"**Sharpe:** {worst['sharpe']:.2f}")
        st.write(f"**Return:** {worst['total_return']:.2f}%")
    
    with summary_col3:
        st.subheader("Overall Statistics")
        st.write(f"**Strategies Tested:** {len(df)}")
        st.write(f"**Avg Sharpe:** {df['sharpe'].mean():.2f}")
        st.write(f"**Avg Return:** {df['total_return'].mean():.2f}%")
        st.write(f"**Max Return:** {df['total_return'].max():.2f}%")
        st.write(f"**Min Return:** {df['total_return'].min():.2f}%")
    
    st.subheader("Top 10 Strategies by Sharpe Ratio")
    top10 = df.nlargest(10, 'sharpe')[['spread_method', 'z_window', 'entry', 'exit', 
                                        'stop_loss', 'sharpe', 'total_return', 'max_dd', 
                                        'win_rate', 'num_trades']]
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
