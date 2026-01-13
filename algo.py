from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import time
import json
import os
import asyncio
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class IBPairsTradingBot:
    """
    Automated pairs trading bot using Interactive Brokers with ib_insync.
    """
    
    def __init__(self, ticker1, ticker2, entry_threshold, exit_threshold, 
                 stop_loss, z_window, spread_method='hedge_ratio',
                 check_interval=300, port=7497, client_id=1):
        """
        Initialize the IB trading bot.
        
        Parameters:
        -----------
        ticker1, ticker2 : str
            Stock ticker symbols
        entry_threshold : float
            Z-score to enter position
        exit_threshold : float
            Z-score to exit position
        stop_loss : float
            Z-score for stop loss
        z_window : int
            Rolling window for z-score calculation
        spread_method : str
            'hedge_ratio', 'log_ratio', or 'ratio'
        check_interval : int
            Seconds between checks (default: 300 = 5 minutes)
        port : int
            IB Gateway/TWS port (7497=TWS Paper, 7496=TWS Live, 4002=Gateway Paper, 4001=Gateway Live)
        client_id : int
            Unique client ID for this connection
        """
        self.ticker1 = ticker1.upper()
        self.ticker2 = ticker2.upper()
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.z_window = z_window
        self.hedge_ratio_window = 252  # FIXED: Always use 252 days for hedge ratio
        self.spread_method = spread_method
        self.check_interval = check_interval
        self.port = port
        self.client_id = client_id
        
        # IB connection
        self.ib = IB()
        self.connected = False
        
        # Contract objects
        self.contract1 = None
        self.contract2 = None
        
        # Trading state
        self.current_position = 0  # 0: flat, 1: long spread, -1: short spread
        self.position_size = 100  # Number of shares for ticker1
        self.entry_price1 = None
        self.entry_price2 = None
        self.entry_time = None
        
        # Data storage
        self.historical_data = None
        self.trade_log = []
        self.signal_log = []
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        self.log_file = f"logs/ib_bot_{self.ticker1}_{self.ticker2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log_message(self, message, level="INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def connect(self, host='127.0.0.1'):
        """Connect to Interactive Brokers."""
        try:
            self.log_message(f"Connecting to IB on {host}:{self.port}...")
            self.ib.connect(host, self.port, clientId=self.client_id)
            self.connected = True
            self.log_message("Successfully connected to IB!")
            
            # Note: We don't request market data type as we use historical snapshots
            self.log_message("Using historical snapshots for price data (no subscription required)")
            
            # Create contract objects
            self.contract1 = Stock(self.ticker1, 'SMART', 'USD')
            self.contract2 = Stock(self.ticker2, 'SMART', 'USD')
            
            # Qualify contracts
            self.ib.qualifyContracts(self.contract1, self.contract2)
            self.log_message(f"Qualified contracts: {self.ticker1}, {self.ticker2}")
            
            # Get account info
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    self.log_message(f"Account Net Liquidation: {av.value} {av.currency}")
                    break
            
            return True
            
        except Exception as e:
            self.log_message(f"Failed to connect to IB: {e}", "ERROR")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.log_message("Disconnected from IB")
    
    def get_live_prices(self):
        """Get current prices from IB using historical snapshot (no subscription needed)."""
        try:
            self.log_message("Fetching current prices via historical snapshot...")
            
            # Request recent data - use 1 day to ensure we get data even when market is closed
            # This will give us the most recent closing prices
            bars1 = self.ib.reqHistoricalData(
                self.contract1,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            bars2 = self.ib.reqHistoricalData(
                self.contract2,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if bars1 and bars2 and len(bars1) > 0 and len(bars2) > 0:
                price1 = bars1[-1].close
                price2 = bars2[-1].close
                bar_time1 = bars1[-1].date
                bar_time2 = bars2[-1].date
                self.log_message(f"Prices - {self.ticker1}: ${price1:.2f} (from {bar_time1}), {self.ticker2}: ${price2:.2f} (from {bar_time2})")
                return float(price1), float(price2), datetime.now()
            else:
                self.log_message("Unable to get prices from snapshot", "ERROR")
                return None, None, None
                
        except Exception as e:
            self.log_message(f"Error getting prices: {e}", "ERROR")
            return None, None, None
    
    def fetch_historical_data(self, days=365):
        """Fetch historical data from IB."""
        try:
            # Fetch at least hedge_ratio_window + some buffer
            days_needed = max(days, self.hedge_ratio_window + 30)
            self.log_message(f"Fetching {days_needed} days of historical data...")
            
            # Request historical data
            end_datetime = datetime.now()
            duration = f"{days_needed} D"
            
            bars1 = self.ib.reqHistoricalData(
                self.contract1,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            bars2 = self.ib.reqHistoricalData(
                self.contract2,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            # Convert to DataFrame
            df1 = util.df(bars1)
            df2 = util.df(bars2)
            
            if df1.empty or df2.empty:
                self.log_message("No historical data received", "ERROR")
                return None
            
            self.historical_data = pd.DataFrame({
                self.ticker1: df1.set_index('date')['close'],
                self.ticker2: df2.set_index('date')['close']
            }).dropna()
            
            self.log_message(f"Loaded {len(self.historical_data)} days of historical data")
            self.log_message(f"Using {self.hedge_ratio_window}-day window for hedge ratio, {self.z_window}-day window for z-score")
            return self.historical_data
            
        except Exception as e:
            self.log_message(f"Error fetching historical data: {e}", "ERROR")
            return None
    
    def get_current_positions(self):
        """Get current positions from IB."""
        positions = self.ib.positions()
        pos_dict = {}
        
        for pos in positions:
            if pos.contract.symbol == self.ticker1:
                pos_dict[self.ticker1] = pos.position
            elif pos.contract.symbol == self.ticker2:
                pos_dict[self.ticker2] = pos.position
        
        return pos_dict
    
    def place_order(self, contract, action, quantity, order_type='MKT'):
        """Place order with IB."""
        try:
            if order_type == 'MKT':
                order = MarketOrder(action, quantity)
            else:
                # Could add limit orders here
                order = MarketOrder(action, quantity)
            
            trade = self.ib.placeOrder(contract, order)
            self.log_message(f"Order placed: {action} {quantity} {contract.symbol}")
            
            # Wait for order to fill
            timeout = 30
            start_time = time.time()
            while not trade.isDone():
                self.ib.sleep(1)
                if time.time() - start_time > timeout:
                    self.log_message("Order timeout - canceling", "WARNING")
                    self.ib.cancelOrder(order)
                    return None
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                self.log_message(f"Order filled at ${fill_price:.2f}")
                return fill_price
            else:
                self.log_message(f"Order status: {trade.orderStatus.status}", "WARNING")
                return None
                
        except Exception as e:
            self.log_message(f"Error placing order: {e}", "ERROR")
            return None
    
    def execute_long_spread_entry(self, price1, price2, hedge_ratio):
        """Enter long spread position: Buy ticker1, Sell ticker2."""
        self.log_message("EXECUTING LONG SPREAD ENTRY")
        
        # Calculate quantities
        qty1 = self.position_size
        qty2 = int(self.position_size * hedge_ratio)
        
        self.log_message(f"  Hedge Ratio: {hedge_ratio:.4f}")
        self.log_message(f"  BUY {qty1} shares of {self.ticker1}")
        self.log_message(f"  SELL {qty2} shares of {self.ticker2}")
        
        # Place orders
        fill1 = self.place_order(self.contract1, 'BUY', qty1)
        fill2 = self.place_order(self.contract2, 'SELL', qty2)
        
        if fill1 and fill2:
            self.current_position = 1
            self.entry_price1 = fill1
            self.entry_price2 = fill2
            self.entry_time = datetime.now()
            
            self.log_message("LONG SPREAD ENTRY COMPLETED")
            return True
        else:
            self.log_message("LONG SPREAD ENTRY FAILED", "ERROR")
            return False
    
    def execute_short_spread_entry(self, price1, price2, hedge_ratio):
        """Enter short spread position: Sell ticker1, Buy ticker2."""
        self.log_message("EXECUTING SHORT SPREAD ENTRY")
        
        # Calculate quantities
        qty1 = self.position_size
        qty2 = int(self.position_size * hedge_ratio)
        
        self.log_message(f"  Hedge Ratio: {hedge_ratio:.4f}")
        self.log_message(f"  SELL {qty1} shares of {self.ticker1}")
        self.log_message(f"  BUY {qty2} shares of {self.ticker2}")
        
        # Place orders
        fill1 = self.place_order(self.contract1, 'SELL', qty1)
        fill2 = self.place_order(self.contract2, 'BUY', qty2)
        
        if fill1 and fill2:
            self.current_position = -1
            self.entry_price1 = fill1
            self.entry_price2 = fill2
            self.entry_time = datetime.now()
            
            self.log_message("SHORT SPREAD ENTRY COMPLETED")
            return True
        else:
            self.log_message("SHORT SPREAD ENTRY FAILED", "ERROR")
            return False
    
    def execute_exit(self, reason="EXIT"):
        """Close all positions."""
        self.log_message(f"EXECUTING {reason}")
        
        # Get current positions
        positions = self.get_current_positions()
        
        # Close positions
        success = True
        for ticker, qty in positions.items():
            if qty != 0:
                contract = self.contract1 if ticker == self.ticker1 else self.contract2
                action = 'SELL' if qty > 0 else 'BUY'
                fill = self.place_order(contract, action, abs(qty))
                if not fill:
                    success = False
        
        if success:
            self.log_message(f"{reason} COMPLETED")
            
            # Calculate P&L
            if self.entry_price1 and self.entry_price2:
                current_positions = self.get_current_positions()
                # P&L calculation would be based on actual fill prices
                self.log_message("Position closed successfully")
            
            self.current_position = 0
            self.entry_price1 = None
            self.entry_price2 = None
            self.entry_time = None
            return True
        else:
            self.log_message(f"{reason} FAILED", "ERROR")
            return False
    
    def generate_signal(self):
        """Generate trading signal based on current market conditions."""
        price1, price2, timestamp = self.get_live_prices()
        
        if price1 is None or price2 is None:
            return None
        
        # IMPORTANT: Calculate hedge ratio BEFORE adding live prices
        # This prevents the hedge ratio from shifting with each new price point
        if self.spread_method == 'hedge_ratio':
            # Calculate hedge ratio using existing historical data only
            window_data = self.historical_data.iloc[-self.hedge_ratio_window:]
            slope, _, _, _, _ = stats.linregress(
                window_data[self.ticker2], window_data[self.ticker1]
            )
            hedge_ratio = slope
        else:
            hedge_ratio = 1.0
        
        # Now create temporary dataset with live price for z-score calculation
        temp_data = self.historical_data.copy()
        new_row = pd.DataFrame({
            self.ticker1: [price1],
            self.ticker2: [price2]
        }, index=[timestamp])
        temp_data = pd.concat([temp_data, new_row])
        
        # Calculate spread using the fixed hedge ratio
        if self.spread_method == 'ratio':
            spread = temp_data[self.ticker1] / temp_data[self.ticker2]
        elif self.spread_method == 'log_ratio':
            spread = np.log(temp_data[self.ticker1]) - np.log(temp_data[self.ticker2])
        elif self.spread_method == 'hedge_ratio':
            spread = temp_data[self.ticker1] - (hedge_ratio * temp_data[self.ticker2])
        else:
            spread = temp_data[self.ticker1] - temp_data[self.ticker2]
        
        # Calculate z-score using z_window
        spread_mean = spread.iloc[-self.z_window:].mean()
        spread_std = spread.iloc[-self.z_window:].std()
        
        if spread_std == 0:
            return None
        
        z_score = (spread.iloc[-1] - spread_mean) / spread_std
        
        # Check if prices changed significantly to determine if we should update historical data
        data_updated = False
        if len(self.historical_data) > 0:
            last_price1 = self.historical_data[self.ticker1].iloc[-1]
            last_price2 = self.historical_data[self.ticker2].iloc[-1]
            
            # Check for significant price change (> 0.01%)
            price_change1 = abs(price1 - last_price1) / last_price1
            price_change2 = abs(price2 - last_price2) / last_price2
            
            if price_change1 >= 0.0001 or price_change2 >= 0.0001:
                # Update historical data for next iteration
                self.historical_data = pd.concat([self.historical_data, new_row])
                
                # Keep only recent data to manage memory
                keep_rows = max(1000, self.hedge_ratio_window + 100)
                if len(self.historical_data) > keep_rows:
                    self.historical_data = self.historical_data.iloc[-keep_rows:]
                
                data_updated = True
                self.log_message("Updated historical data with new prices")
        else:
            # First data point
            self.historical_data = pd.concat([self.historical_data, new_row])
            data_updated = True
        
        signal = {
            'timestamp': timestamp,
            'price1': price1,
            'price2': price2,
            'z_score': z_score,
            'spread': spread.iloc[-1],
            'hedge_ratio': hedge_ratio,
            'current_position': self.current_position,
            'action': 'HOLD',
            'data_updated': data_updated
        }
        
        # Determine action
        if self.current_position != 0 and abs(z_score) > self.stop_loss:
            signal['action'] = 'STOP_LOSS'
        elif self.current_position == 0:
            if z_score < -self.entry_threshold:
                signal['action'] = 'LONG_ENTRY'
            elif z_score > self.entry_threshold:
                signal['action'] = 'SHORT_ENTRY'
        elif abs(z_score) < self.exit_threshold:
            signal['action'] = 'EXIT'
        
        status = "NEW DATA" if data_updated else "UNCHANGED"
        self.log_message(f"Signal: {signal['action']}, Z-Score: {z_score:.2f}, Hedge Ratio: {hedge_ratio:.4f} [{status}]")
        self.signal_log.append(signal)
        
        return signal
    
    def execute_signal(self, signal):
        """Execute trading signal."""
        action = signal['action']
        
        if action == 'HOLD':
            return
        
        try:
            if action == 'LONG_ENTRY':
                self.execute_long_spread_entry(
                    signal['price1'], signal['price2'], signal['hedge_ratio']
                )
            elif action == 'SHORT_ENTRY':
                self.execute_short_spread_entry(
                    signal['price1'], signal['price2'], signal['hedge_ratio']
                )
            elif action in ['EXIT', 'STOP_LOSS']:
                self.execute_exit(reason=action)
                
        except Exception as e:
            self.log_message(f"Error executing signal: {e}", "ERROR")
    
    def run(self, duration_hours=None):
        """Run the trading bot."""
        self.log_message("="*60)
        self.log_message("STARTING IB PAIRS TRADING BOT")
        self.log_message("="*60)
        self.log_message(f"Pair: {self.ticker1} / {self.ticker2}")
        self.log_message(f"Entry: ±{self.entry_threshold}σ, Exit: ±{self.exit_threshold}σ, Stop: ±{self.stop_loss}σ")
        self.log_message(f"Hedge Ratio Window: {self.hedge_ratio_window} days (FIXED)")
        self.log_message(f"Z-Score Window: {self.z_window} days")
        self.log_message(f"Spread Method: {self.spread_method}")
        self.log_message(f"Port: {self.port} ({'PAPER' if self.port in [7497, 4002] else 'LIVE'})")
        self.log_message("="*60)
        
        # Connect to IB
        if not self.connect():
            self.log_message("Failed to connect to IB. Exiting.", "ERROR")
            return
        
        # Fetch historical data
        if self.fetch_historical_data() is None:
            self.log_message("Failed to fetch historical data. Exiting.", "ERROR")
            self.disconnect()
            return
        
        start_time = datetime.now()
        iteration = 0
        
        try:
            while True:
                iteration += 1
                self.log_message(f"\n--- Iteration {iteration} ---")
                
                # Generate and execute signal
                signal = self.generate_signal()
                if signal:
                    # Only execute trades during market hours for real trading
                    if signal['action'] != 'HOLD':
                        if self._is_market_hours():
                            self.execute_signal(signal)
                        else:
                            self.log_message("Outside market hours - signal generated but not executed", "INFO")
                            self.log_message(f"Signal would be: {signal['action']}", "INFO")
                
                # Check duration
                if duration_hours is not None:
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        self.log_message(f"Duration limit of {duration_hours} hours reached")
                        break
                
                # Wait for next check
                self.log_message(f"Next check in {self.check_interval} seconds...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.log_message("\nBot stopped by user")
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """Clean shutdown."""
        self.log_message("\n" + "="*60)
        self.log_message("SHUTTING DOWN BOT")
        self.log_message("="*60)
        
        # Check for open positions
        if self.current_position != 0:
            self.log_message("WARNING: Open position detected!", "WARNING")
            close = input("Close all positions before shutdown? (y/n): ").lower()
            if close == 'y':
                self.execute_exit(reason="SHUTDOWN")
        
        self.disconnect()
        self.log_message("Bot shutdown complete")
    
    def _is_market_hours(self):
        """Check if current time is during US market hours (9:30 AM - 4:00 PM ET)."""
        import pytz
        
        # Get current time in ET
        et_tz = pytz.timezone('US/Eastern')
        current_time_et = datetime.now(et_tz)
        
        # Check if weekday
        if current_time_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if during market hours (9:30 AM - 4:00 PM ET)
        market_open = current_time_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= current_time_et <= market_close


def start_ib_bot():
    """Interactive function to start the IB bot."""
    print("\n" + "="*60)
    print("IB PAIRS TRADING BOT - CONFIGURATION")
    print("="*60)
    
    # Get user inputs
    ticker1 = input("\nEnter first ticker: ").upper()
    ticker2 = input("Enter second ticker: ").upper()
    
    print("\n--- Strategy Parameters ---")
    entry_threshold = float(input("Entry threshold (σ) [2.0]: ") or "2.0")
    exit_threshold = float(input("Exit threshold (σ) [0.5]: ") or "0.5")
    stop_loss = float(input("Stop loss (σ) [3.0]: ") or "3.0")
    z_window = int(input("Z-score window (days) [60]: ") or "60")
    
    print("\n--- Spread Method ---")
    print("1. Hedge Ratio (252-day window)")
    print("2. Log Ratio")
    print("3. Simple Ratio")
    method = input("Select [1]: ") or "1"
    methods = {'1': 'hedge_ratio', '2': 'log_ratio', '3': 'ratio'}
    spread_method = methods.get(method, 'hedge_ratio')
    
    print("\n--- IB Connection ---")
    print("Port options:")
    print("  7497 = TWS Paper Trading")
    print("  7496 = TWS Live Trading")
    print("  4002 = IB Gateway Paper Trading")
    print("  4001 = IB Gateway Live Trading")
    port = int(input("Port [7497]: ") or "7497")
    
    print("\n--- Bot Settings ---")
    check_interval = int(input("Check interval (seconds) [300]: ") or "300")
    position_size = int(input("Position size (shares) [100]: ") or "100")
    
    # Confirm
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Pair: {ticker1} / {ticker2}")
    print(f"Entry: ±{entry_threshold}σ, Exit: ±{exit_threshold}σ, Stop: ±{stop_loss}σ")
    print(f"Spread Method: {spread_method}")
    if spread_method == 'hedge_ratio':
        print(f"Hedge Ratio Window: 252 days (fixed)")
    print(f"Z-Score Window: {z_window} days")
    print(f"Port: {port} ({'PAPER' if port in [7497, 4002] else 'LIVE TRADING'})")
    print(f"Position Size: {position_size} shares")
    print("="*60)
    
    if port not in [7497, 4002]:
        print("\n⚠️  WARNING: YOU ARE USING LIVE TRADING PORT!")
        confirm = input("Type 'CONFIRM' to proceed with LIVE trading: ")
        if confirm != 'CONFIRM':
            print("Cancelled")
            return None
    
    confirm = input("\nStart bot? (y/n): ").lower()
    
    if confirm == 'y':
        bot = IBPairsTradingBot(
            ticker1=ticker1,
            ticker2=ticker2,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss=stop_loss,
            z_window=z_window,
            spread_method=spread_method,
            check_interval=check_interval,
            port=port
        )
        bot.position_size = position_size
        bot.run()
        return bot
    else:
        print("Cancelled")
        return None


if __name__ == "__main__":
    # Make sure IB Gateway or TWS is running first!
    print("\n⚠️  IMPORTANT: Make sure IB Gateway or TWS is running and logged in!")
    print("API settings must be enabled in TWS/Gateway configuration.\n")
    print("NOTE: Hedge ratio calculations use a fixed 252-day window for stability.")
    print("      Z-score calculations use the configurable window you specify.\n")
    
    input("Press Enter when ready...")
    
    bot = start_ib_bot()
