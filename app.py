"""
High-Frequency Trading Signal Generator
Advanced market microstructure analysis and signal generation for proprietary trading
Author: Spencer Purdy
"""

# Install required packages
# !pip install gradio numpy pandas matplotlib seaborn plotly scipy scikit-learn -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from scipy import stats
from datetime import datetime, timedelta
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class OrderBook:
    """
    Simulated limit order book with realistic market microstructure dynamics.
    """
    
    def __init__(self, ticker, initial_price=100.0, tick_size=0.01):
        self.ticker = ticker
        self.tick_size = tick_size
        self.last_price = initial_price
        self.timestamp = 0
        
        # Order book structure: price -> quantity
        self.bids = {}  # Buy orders
        self.asks = {}  # Sell orders
        
        # Initialize with realistic order book
        self._initialize_book()
        
        # Track order book events
        self.events = []
        
    def _initialize_book(self):
        """Initialize order book with realistic depth and spread."""
        # Generate initial bid/ask levels
        spread = np.random.uniform(0.01, 0.03)
        
        # Bids
        for i in range(10):
            price = round(self.last_price - spread/2 - i * self.tick_size, 2)
            quantity = np.random.randint(100, 1000) * 100
            self.bids[price] = quantity
            
        # Asks
        for i in range(10):
            price = round(self.last_price + spread/2 + i * self.tick_size, 2)
            quantity = np.random.randint(100, 1000) * 100
            self.asks[price] = quantity
    
    def get_best_bid_ask(self):
        """Get current best bid and ask prices with sizes."""
        if not self.bids or not self.asks:
            return None, None, None, None
            
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        
        return best_bid, self.bids[best_bid], best_ask, self.asks[best_ask]
    
    def get_mid_price(self):
        """Calculate mid price."""
        best_bid, _, best_ask, _ = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return self.last_price
    
    def get_spread(self):
        """Calculate bid-ask spread."""
        best_bid, _, best_ask, _ = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return 0
    
    def get_book_depth(self, levels=5):
        """Get order book depth up to specified levels."""
        sorted_bids = sorted(self.bids.items(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.items())[:levels]
        
        return sorted_bids, sorted_asks
    
    def simulate_tick(self):
        """Simulate one tick of order book activity."""
        self.timestamp += 1
        
        # Random events: new orders, cancellations, trades
        event_type = np.random.choice(['new_order', 'cancel', 'trade'], 
                                    p=[0.5, 0.3, 0.2])
        
        if event_type == 'new_order':
            self._add_random_order()
        elif event_type == 'cancel':
            self._cancel_random_order()
        else:
            self._simulate_trade()
            
    def _add_random_order(self):
        """Add a random limit order to the book."""
        side = np.random.choice(['bid', 'ask'])
        
        if side == 'bid':
            # Price within 10 ticks of best bid
            best_bid = max(self.bids.keys()) if self.bids else self.last_price - 0.01
            price = round(best_bid - np.random.randint(0, 10) * self.tick_size, 2)
            quantity = np.random.randint(1, 20) * 100
            
            if price in self.bids:
                self.bids[price] += quantity
            else:
                self.bids[price] = quantity
                
        else:
            # Price within 10 ticks of best ask
            best_ask = min(self.asks.keys()) if self.asks else self.last_price + 0.01
            price = round(best_ask + np.random.randint(0, 10) * self.tick_size, 2)
            quantity = np.random.randint(1, 20) * 100
            
            if price in self.asks:
                self.asks[price] += quantity
            else:
                self.asks[price] = quantity
    
    def _cancel_random_order(self):
        """Cancel a random order from the book."""
        side = np.random.choice(['bid', 'ask'])
        
        if side == 'bid' and self.bids:
            price = np.random.choice(list(self.bids.keys()))
            cancel_size = min(self.bids[price], np.random.randint(1, 10) * 100)
            self.bids[price] -= cancel_size
            if self.bids[price] <= 0:
                del self.bids[price]
                
        elif self.asks:
            price = np.random.choice(list(self.asks.keys()))
            cancel_size = min(self.asks[price], np.random.randint(1, 10) * 100)
            self.asks[price] -= cancel_size
            if self.asks[price] <= 0:
                del self.asks[price]
    
    def _simulate_trade(self):
        """Simulate a trade crossing the spread."""
        best_bid, bid_size, best_ask, ask_size = self.get_best_bid_ask()
        
        if not best_bid or not best_ask:
            return
            
        # Aggressive order takes liquidity
        if np.random.random() > 0.5:  # Buy market order
            trade_size = min(ask_size, np.random.randint(1, 10) * 100)
            self.asks[best_ask] -= trade_size
            if self.asks[best_ask] <= 0:
                del self.asks[best_ask]
            self.last_price = best_ask
            
        else:  # Sell market order
            trade_size = min(bid_size, np.random.randint(1, 10) * 100)
            self.bids[best_bid] -= trade_size
            if self.bids[best_bid] <= 0:
                del self.bids[best_bid]
            self.last_price = best_bid

class HFTSignalGenerator:
    """
    High-frequency trading signal generator implementing multiple market microstructure signals.
    """
    
    def __init__(self):
        self.order_book = None
        self.signal_history = []
        self.trades = []
        self.performance_metrics = {}
        
    def initialize_market(self, ticker, initial_price):
        """Initialize market with order book."""
        self.order_book = OrderBook(ticker, initial_price)
        
    def calculate_order_flow_imbalance(self, levels=5):
        """
        Calculate order flow imbalance signal.
        
        OFI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        Range: [-1, 1], where positive indicates buying pressure
        """
        bid_levels, ask_levels = self.order_book.get_book_depth(levels)
        
        total_bid_volume = sum(qty for _, qty in bid_levels)
        total_ask_volume = sum(qty for _, qty in ask_levels)
        
        if total_bid_volume + total_ask_volume == 0:
            return 0
            
        ofi = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        return ofi
    
    def calculate_price_pressure(self, levels=3):
        """
        Calculate weighted price pressure indicator.
        
        Measures the average distance of orders from mid price, weighted by volume.
        """
        mid_price = self.order_book.get_mid_price()
        bid_levels, ask_levels = self.order_book.get_book_depth(levels)
        
        # Weighted bid pressure
        bid_pressure = 0
        bid_volume = 0
        for price, qty in bid_levels:
            distance = (mid_price - price) / mid_price
            bid_pressure += distance * qty
            bid_volume += qty
            
        # Weighted ask pressure
        ask_pressure = 0
        ask_volume = 0
        for price, qty in ask_levels:
            distance = (price - mid_price) / mid_price
            ask_pressure += distance * qty
            ask_volume += qty
            
        if bid_volume + ask_volume == 0:
            return 0
            
        # Normalize
        total_pressure = (bid_pressure - ask_pressure) / (bid_volume + ask_volume)
        return total_pressure * 100  # Scale for readability
    
    def calculate_microprice(self):
        """
        Calculate microprice - probability weighted price.
        
        Microprice = (Bid * Ask_Size + Ask * Bid_Size) / (Bid_Size + Ask_Size)
        """
        best_bid, bid_size, best_ask, ask_size = self.order_book.get_best_bid_ask()
        
        if not best_bid or not best_ask or (bid_size + ask_size) == 0:
            return self.order_book.get_mid_price()
            
        microprice = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
        return microprice
    
    def calculate_book_imbalance_ratio(self):
        """
        Calculate book imbalance ratio at best quotes.
        
        BIR = Bid_Size / (Bid_Size + Ask_Size)
        Range: [0, 1], where > 0.5 indicates bid pressure
        """
        _, bid_size, _, ask_size = self.order_book.get_best_bid_ask()
        
        if not bid_size or not ask_size:
            return 0.5
            
        return bid_size / (bid_size + ask_size)
    
    def calculate_spread_signal(self):
        """
        Calculate spread-based signal.
        
        Normalized spread relative to typical spread.
        """
        current_spread = self.order_book.get_spread()
        typical_spread = 0.02  # 2 cents typical
        
        if typical_spread == 0:
            return 0
            
        # Negative when spread is tight (good for market making)
        return (current_spread - typical_spread) / typical_spread
    
    def calculate_volume_clock(self, window=100):
        """
        Calculate volume clock signal - time between volume events.
        
        Faster volume clock indicates more activity.
        """
        if len(self.signal_history) < 2:
            return 0
            
        # Simple proxy: changes in order book depth
        recent_changes = len([s for s in self.signal_history[-window:] 
                            if abs(s.get('ofi', 0)) > 0.1])
        
        return recent_changes / window
    
    def generate_composite_signal(self):
        """
        Generate composite HFT signal from multiple indicators.
        
        Returns:
        Dictionary with all signals and composite score
        """
        signals = {
            'timestamp': self.order_book.timestamp,
            'mid_price': self.order_book.get_mid_price(),
            'spread': self.order_book.get_spread(),
            'ofi': self.calculate_order_flow_imbalance(),
            'price_pressure': self.calculate_price_pressure(),
            'microprice': self.calculate_microprice(),
            'book_imbalance': self.calculate_book_imbalance_ratio(),
            'spread_signal': self.calculate_spread_signal(),
            'volume_clock': self.calculate_volume_clock()
        }
        
        # Composite signal with weights
        weights = {
            'ofi': 0.3,
            'price_pressure': 0.2,
            'book_imbalance': 0.2,
            'spread_signal': 0.15,
            'volume_clock': 0.15
        }
        
        # Normalize book imbalance to [-1, 1]
        normalized_book_imbalance = (signals['book_imbalance'] - 0.5) * 2
        
        composite = (
            weights['ofi'] * signals['ofi'] +
            weights['price_pressure'] * signals['price_pressure'] / 10 +  # Scale down
            weights['book_imbalance'] * normalized_book_imbalance +
            weights['spread_signal'] * (-signals['spread_signal']) +  # Invert
            weights['volume_clock'] * (signals['volume_clock'] - 0.5) * 2
        )
        
        signals['composite_signal'] = composite
        
        # Trading decision thresholds
        if composite > 0.3:
            signals['action'] = 'BUY'
        elif composite < -0.3:
            signals['action'] = 'SELL'
        else:
            signals['action'] = 'HOLD'
            
        return signals
    
    def simulate_trading_session(self, duration_seconds=60, ticks_per_second=100):
        """
        Simulate a high-frequency trading session.
        
        Parameters:
        duration_seconds: Length of simulation
        ticks_per_second: Market data frequency
        """
        total_ticks = duration_seconds * ticks_per_second
        
        # Reset tracking
        self.signal_history = []
        self.trades = []
        position = 0
        pnl = 0
        
        # Performance tracking
        entry_price = None
        trade_count = 0
        winning_trades = 0
        total_profit = 0
        
        print(f"Simulating {duration_seconds} seconds at {ticks_per_second} ticks/second...")
        
        for tick in range(total_ticks):
            # Update order book
            self.order_book.simulate_tick()
            
            # Generate signals
            signals = self.generate_composite_signal()
            self.signal_history.append(signals)
            
            # Execute trades based on signals
            current_price = signals['mid_price']
            
            if signals['action'] == 'BUY' and position <= 0:
                if position < 0:  # Close short
                    profit = entry_price - current_price
                    pnl += profit
                    total_profit += profit
                    if profit > 0:
                        winning_trades += 1
                    trade_count += 1
                    
                position = 1
                entry_price = current_price
                self.trades.append({
                    'timestamp': tick,
                    'action': 'BUY',
                    'price': current_price,
                    'position': position
                })
                
            elif signals['action'] == 'SELL' and position >= 0:
                if position > 0:  # Close long
                    profit = current_price - entry_price
                    pnl += profit
                    total_profit += profit
                    if profit > 0:
                        winning_trades += 1
                    trade_count += 1
                    
                position = -1
                entry_price = current_price
                self.trades.append({
                    'timestamp': tick,
                    'action': 'SELL',
                    'price': current_price,
                    'position': position
                })
        
        # Close final position
        if position != 0:
            final_price = self.signal_history[-1]['mid_price']
            if position > 0:
                profit = final_price - entry_price
            else:
                profit = entry_price - final_price
            pnl += profit
            total_profit += profit
            if profit > 0:
                winning_trades += 1
            trade_count += 1
        
        # Calculate performance metrics
        self.performance_metrics = {
            'total_trades': len(self.trades),
            'pnl': pnl,
            'win_rate': winning_trades / trade_count if trade_count > 0 else 0,
            'avg_profit_per_trade': total_profit / trade_count if trade_count > 0 else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown()
        }
        
        return self.performance_metrics
    
    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from signal history."""
        if len(self.signal_history) < 2:
            return 0
            
        returns = []
        for i in range(1, len(self.signal_history)):
            prev_price = self.signal_history[i-1]['mid_price']
            curr_price = self.signal_history[i]['mid_price']
            returns.append((curr_price - prev_price) / prev_price)
            
        if not returns or np.std(returns) == 0:
            return 0
            
        return np.mean(returns) / np.std(returns) * np.sqrt(252 * 6.5 * 3600)  # Annualized
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown from trades."""
        if not self.trades:
            return 0
            
        cumulative_pnl = []
        current_pnl = 0
        
        for i, trade in enumerate(self.trades):
            if i > 0 and i % 2 == 0:  # Every round trip
                prev_trade = self.trades[i-1]
                if trade['action'] == 'SELL':
                    current_pnl += trade['price'] - prev_trade['price']
                else:
                    current_pnl += prev_trade['price'] - trade['price']
            cumulative_pnl.append(current_pnl)
            
        if not cumulative_pnl:
            return 0
            
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (pnl - peak) / peak if peak != 0 else 0
            max_dd = min(max_dd, drawdown)
            
        return abs(max_dd)
    
    def create_visualizations(self):
        """Create comprehensive HFT visualizations."""
        if not self.signal_history:
            return None, None
            
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.signal_history)
        
        # Create main visualization figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Order Flow Imbalance', 'Price & Microprice',
                          'Composite Signal with Actions', 'Spread Analysis',
                          'Book Imbalance Ratio', 'Signal Distribution'),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Order Flow Imbalance
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['ofi'], 
                      mode='lines', name='OFI',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # 2. Price & Microprice
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['mid_price'], 
                      mode='lines', name='Mid Price',
                      line=dict(color='black', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['microprice'], 
                      mode='lines', name='Microprice',
                      line=dict(color='red', width=1, dash='dot')),
            row=1, col=2
        )
        
        # 3. Composite Signal with Trading Actions
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['composite_signal'], 
                      mode='lines', name='Composite Signal',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # Add buy/sell markers
        buy_signals = df[df['action'] == 'BUY']
        sell_signals = df[df['action'] == 'SELL']
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(x=buy_signals['timestamp'], y=buy_signals['composite_signal'],
                          mode='markers', name='BUY',
                          marker=dict(color='green', size=8, symbol='triangle-up')),
                row=2, col=1
            )
            
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(x=sell_signals['timestamp'], y=sell_signals['composite_signal'],
                          mode='markers', name='SELL',
                          marker=dict(color='red', size=8, symbol='triangle-down')),
                row=2, col=1
            )
        
        # Add threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)
        
        # 4. Spread Analysis
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['spread']*10000, 
                      mode='lines', name='Spread (bps)',
                      line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        # 5. Book Imbalance Ratio
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['book_imbalance'], 
                      mode='lines', name='Book Imbalance',
                      fill='tozeroy',
                      line=dict(color='teal', width=1)),
            row=3, col=1
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1)
        
        # 6. Signal Distribution Histogram
        fig.add_trace(
            go.Histogram(x=df['composite_signal'], nbinsx=50,
                        name='Signal Distribution',
                        marker_color='darkblue'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="High-Frequency Trading Signal Analysis",
            title_font_size=20
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (ticks)", row=3)
        fig.update_yaxes(title_text="OFI", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Signal", row=2, col=1)
        fig.update_yaxes(title_text="Spread (bps)", row=2, col=2)
        fig.update_yaxes(title_text="Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=2)
        
        # Create execution analysis figure
        fig2 = go.Figure()
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            
            # Plot price with trade markers
            fig2.add_trace(
                go.Scatter(x=df['timestamp'], y=df['mid_price'],
                          mode='lines', name='Price',
                          line=dict(color='lightgray', width=1))
            )
            
            # Add trade executions
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if not buy_trades.empty:
                fig2.add_trace(
                    go.Scatter(x=buy_trades['timestamp'], y=buy_trades['price'],
                              mode='markers', name='Buy Orders',
                              marker=dict(color='green', size=10, symbol='triangle-up'))
                )
                
            if not sell_trades.empty:
                fig2.add_trace(
                    go.Scatter(x=sell_trades['timestamp'], y=sell_trades['price'],
                              mode='markers', name='Sell Orders',
                              marker=dict(color='red', size=10, symbol='triangle-down'))
                )
        
        fig2.update_layout(
            title="Trade Execution Analysis",
            xaxis_title="Time (ticks)",
            yaxis_title="Price ($)",
            height=400,
            hovermode='x unified'
        )
        
        return fig, fig2

# Initialize the HFT system
hft_system = HFTSignalGenerator()

def run_hft_simulation(ticker, initial_price, duration_seconds, ticks_per_second,
                      ofi_weight, price_pressure_weight, book_imbalance_weight,
                      spread_weight, volume_clock_weight, signal_threshold):
    """
    Run HFT simulation with customizable parameters.
    """
    try:
        # Update signal weights
        HFTSignalGenerator.weights = {
            'ofi': ofi_weight / 100,
            'price_pressure': price_pressure_weight / 100,
            'book_imbalance': book_imbalance_weight / 100,
            'spread_signal': spread_weight / 100,
            'volume_clock': volume_clock_weight / 100
        }
        
        # Initialize market
        hft_system.initialize_market(ticker, initial_price)
        
        # Run simulation
        metrics = hft_system.simulate_trading_session(duration_seconds, ticks_per_second)
        
        # Create visualizations
        fig1, fig2 = hft_system.create_visualizations()
        
        # Format results
        results_text = f"""
        ## 🚀 High-Frequency Trading Simulation Results
        
        ### Performance Metrics
        - **Total Trades**: {metrics['total_trades']}
        - **P&L**: ${metrics['pnl']:.2f}
        - **Win Rate**: {metrics['win_rate']*100:.1f}%
        - **Average Profit per Trade**: ${metrics['avg_profit_per_trade']:.4f}
        - **Sharpe Ratio**: {metrics['sharpe_ratio']:.3f}
        - **Maximum Drawdown**: {metrics['max_drawdown']*100:.1f}%
        
        ### Trading Statistics
        - **Trades per Second**: {metrics['total_trades']/duration_seconds:.1f}
        - **Signal Threshold**: ±{signal_threshold}
        - **Market Data Rate**: {ticks_per_second} ticks/second
        - **Simulation Duration**: {duration_seconds} seconds
        
        ### Signal Weights
        - Order Flow Imbalance: {ofi_weight}%
        - Price Pressure: {price_pressure_weight}%
        - Book Imbalance: {book_imbalance_weight}%
        - Spread Signal: {spread_weight}%
        - Volume Clock: {volume_clock_weight}%
        
        ### Market Microstructure Analysis
        """
        
        # Add microstructure statistics
        if hft_system.signal_history:
            df = pd.DataFrame(hft_system.signal_history)
            
            avg_spread = df['spread'].mean() * 10000  # Convert to basis points
            avg_ofi = df['ofi'].mean()
            signal_volatility = df['composite_signal'].std()
            
            results_text += f"""
        - **Average Spread**: {avg_spread:.2f} basis points
        - **Average Order Flow Imbalance**: {avg_ofi:.4f}
        - **Signal Volatility**: {signal_volatility:.4f}
        - **Price Range**: ${df['mid_price'].min():.2f} - ${df['mid_price'].max():.2f}
        
        ### Execution Quality
        - **Fill Rate**: {(metrics['total_trades']/(duration_seconds*ticks_per_second))*100:.2f}%
        - **Latency Assumption**: < 1 microsecond
        - **Market Impact**: Minimal (simulated)
        """
        
        return results_text, fig1, fig2
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="HFT Signal Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # ⚡ High-Frequency Trading Signal Generator
        
        ### Advanced Market Microstructure Analysis & Signal Generation
        Designed for proprietary trading firms and market makers
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Market Configuration")
                ticker = gr.Textbox(label="Ticker Symbol", value="AAPL", placeholder="AAPL")
                initial_price = gr.Number(label="Initial Price", value=100.0)
                
                gr.Markdown("### ⏱️ Simulation Parameters")
                duration_seconds = gr.Slider(
                    label="Duration (seconds)",
                    minimum=10, maximum=300, value=60, step=10,
                    info="Length of trading session"
                )
                ticks_per_second = gr.Slider(
                    label="Market Data Rate (ticks/second)",
                    minimum=10, maximum=1000, value=100, step=10,
                    info="Frequency of order book updates"
                )
                
                gr.Markdown("### 🎯 Signal Weights (%)")
                ofi_weight = gr.Slider(
                    label="Order Flow Imbalance Weight",
                    minimum=0, maximum=100, value=30, step=5
                )
                price_pressure_weight = gr.Slider(
                    label="Price Pressure Weight",
                    minimum=0, maximum=100, value=20, step=5
                )
                book_imbalance_weight = gr.Slider(
                    label="Book Imbalance Weight",
                    minimum=0, maximum=100, value=20, step=5
                )
                spread_weight = gr.Slider(
                    label="Spread Signal Weight",
                    minimum=0, maximum=100, value=15, step=5
                )
                volume_clock_weight = gr.Slider(
                    label="Volume Clock Weight",
                    minimum=0, maximum=100, value=15, step=5
                )
                
                gr.Markdown("### 🎛️ Trading Parameters")
                signal_threshold = gr.Slider(
                    label="Signal Threshold",
                    minimum=0.1, maximum=0.5, value=0.3, step=0.05,
                    info="Minimum signal strength to trigger trades"
                )
                
                simulate_btn = gr.Button("⚡ Run HFT Simulation", variant="primary", scale=2)
        
        with gr.Row():
            results_output = gr.Markdown(label="Results")
        
        with gr.Row():
            signal_chart = gr.Plot(label="Signal Analysis")
        
        with gr.Row():
            execution_chart = gr.Plot(label="Execution Analysis")
        
        # Set up event handler
        simulate_btn.click(
            fn=run_hft_simulation,
            inputs=[ticker, initial_price, duration_seconds, ticks_per_second,
                   ofi_weight, price_pressure_weight, book_imbalance_weight,
                   spread_weight, volume_clock_weight, signal_threshold],
            outputs=[results_output, signal_chart, execution_chart]
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ["AAPL", 100.0, 60, 100, 30, 20, 20, 15, 15, 0.3],
                ["SPY", 450.0, 120, 200, 40, 15, 25, 10, 10, 0.25],
                ["TSLA", 200.0, 30, 500, 25, 25, 15, 20, 15, 0.35],
            ],
            inputs=[ticker, initial_price, duration_seconds, ticks_per_second,
                   ofi_weight, price_pressure_weight, book_imbalance_weight,
                   spread_weight, volume_clock_weight, signal_threshold],
        )
        
        gr.Markdown("""
        ---
        ### 📚 Signal Components
        
        **Order Flow Imbalance (OFI)**
        - Measures the relative volume between bid and ask sides
        - Range: [-1, 1], positive indicates buying pressure
        - Key signal for directional moves
        
        **Price Pressure**
        - Weighted average distance of orders from mid price
        - Indicates where liquidity is concentrated
        - Helps predict short-term price movements
        
        **Microprice**
        - Volume-weighted price between best bid and ask
        - More accurate than mid price for predicting next trade
        - Essential for optimal execution
        
        **Book Imbalance Ratio**
        - Ratio of bid size to total size at best quotes
        - Simple but effective predictor of price direction
        - Range: [0, 1], >0.5 indicates bid pressure
        
        **Spread Signal**
        - Normalized spread relative to typical spread
        - Identifies favorable market making opportunities
        - Negative values indicate tight spreads
        
        **Volume Clock**
        - Measures the rate of order book activity
        - Higher values indicate more trading opportunities
        - Adapts to market regime changes
        
        ### ⚡ HFT Characteristics
        
        - **Ultra-low latency**: Microsecond decision making
        - **Market neutral**: Profits from microstructure inefficiencies
        - **High turnover**: Hundreds of trades per minute
        - **Small edge**: Profits measured in basis points
        - **Technology dependent**: Requires cutting-edge infrastructure
        
        Built for quantitative trading firms and market makers.
        """)
        
    return interface

# Launch the application
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(debug=True, share=True)