---
title: High-Frequency Trading Signal Generator
emoji: ⚡
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
pinned: false
license: mit
short_description: Generate HFT signals with market microstructure analysis.
---

# High-Frequency Trading Signal Generator

An advanced market microstructure analysis platform that generates high-frequency trading signals through sophisticated order book simulation and multi-factor signal generation. This system demonstrates professional-grade HFT strategies by implementing realistic limit order book dynamics, calculating multiple microstructure indicators, and executing simulated trades based on composite signals—all with comprehensive performance tracking and visualization.

## Overview

This project implements a complete high-frequency trading simulation environment that models real market microstructure. It creates a realistic limit order book with bid-ask spreads, simulates market events (new orders, cancellations, trades), generates multiple trading signals from order book data, and executes trades based on a weighted composite signal. The system is designed to demonstrate the complexity of modern HFT strategies while providing educational insights into market microstructure.

## Key Features

### Order Book Simulation
- **Realistic Limit Order Book**: Full depth order book with multiple price levels
- **Dynamic Market Events**: Simulates new orders, cancellations, and trades
- **Tick-by-Tick Updates**: Configurable market data frequency up to 1000 ticks/second
- **Spread Dynamics**: Realistic bid-ask spread behavior
- **Price Discovery**: Natural price movement through order matching

### Trading Signals

**Order Flow Imbalance (OFI)**
- Measures relative volume between bid and ask sides
- Range: [-1, 1] where positive values indicate buying pressure
- Key predictor for short-term directional moves

**Price Pressure Indicator**
- Weighted average distance of orders from mid price
- Indicates where liquidity is concentrated
- Scaled for interpretability

**Microprice**
- Volume-weighted price between best bid and ask
- More accurate than mid price for next trade prediction
- Essential for optimal execution timing

**Book Imbalance Ratio**
- Ratio of bid size to total size at best quotes
- Simple but effective directional predictor
- Range: [0, 1] where >0.5 indicates bid pressure

**Spread Signal**
- Normalized spread relative to typical spread
- Identifies favorable market making opportunities
- Negative values indicate tight spreads

**Volume Clock**
- Measures rate of order book activity
- Adapts to changing market regimes
- Higher values indicate more trading opportunities

### Trading Engine
- **Composite Signal Generation**: Weighted combination of all indicators
- **Configurable Weights**: Adjust importance of each signal component
- **Threshold-Based Execution**: Customizable signal strength requirements
- **Position Management**: Tracks long/short positions with proper entry/exit
- **Real-Time P&L Tracking**: Monitors profit and loss throughout simulation

### Performance Analytics
- **Comprehensive Metrics**: Total trades, P&L, win rate, average profit per trade
- **Risk Metrics**: Sharpe ratio and maximum drawdown calculations
- **Execution Statistics**: Trades per second and fill rates
- **Market Microstructure Analysis**: Average spread, order flow statistics
- **Signal Quality Metrics**: Signal volatility and distribution analysis

## Technologies Used

- **Gradio**: Interactive web interface with real-time parameter adjustment
- **NumPy & Pandas**: High-performance numerical computing and data manipulation
- **Plotly**: Interactive visualizations for signal and execution analysis
- **Matplotlib & Seaborn**: Additional plotting capabilities
- **SciPy**: Statistical computations for performance metrics
- **Python 3.7+**: Core programming language

## Running the Application

### On Hugging Face Spaces
The application is deployed and ready to use at this Hugging Face Space:
1. Access the space through the provided URL
2. Configure market parameters and signal weights
3. Click "Run HFT Simulation" to start
4. Analyze results through interactive charts and metrics

### Local Installation
To run locally:

```bash
# Clone the repository
git clone [your-repo-url]
cd hft-signal-generator

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will launch at `http://localhost:7860`

## Usage Guide

### Market Configuration
1. **Ticker Symbol**: Enter the symbol for simulation (e.g., AAPL, SPY)
2. **Initial Price**: Set the starting price for the order book
3. **Duration**: Choose simulation length (10-300 seconds)
4. **Market Data Rate**: Set ticks per second (10-1000)

### Signal Weight Configuration
Adjust the relative importance of each signal component:
- Order Flow Imbalance (default: 30%)
- Price Pressure (default: 20%)
- Book Imbalance (default: 20%)
- Spread Signal (default: 15%)
- Volume Clock (default: 15%)

Total weights should sum to 100% for optimal performance.

### Trading Parameters
- **Signal Threshold**: Minimum composite signal strength to trigger trades
- Range: 0.1 to 0.5 (default: 0.3)
- Lower values = more trades, higher values = more selective

### Interpreting Results

**Performance Metrics**
- **Total Trades**: Number of round-trip trades executed
- **P&L**: Total profit and loss in currency units
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline

**Signal Analysis Chart**
- Top row: Individual signal components over time
- Middle row: Composite signal with buy/sell markers
- Bottom row: Statistical distributions and patterns

**Execution Analysis Chart**
- Price chart with trade execution points
- Green triangles: Buy orders
- Red triangles: Sell orders
- Gray line: Mid price evolution

## Example Configurations

### Aggressive Market Making
- High spread weight (30-40%)
- Lower signal threshold (0.2)
- Focus on tight spread opportunities

### Directional Trading
- High OFI weight (40-50%)
- Higher signal threshold (0.4)
- Focus on order flow imbalances

### Balanced Approach
- Equal weights across signals
- Medium threshold (0.3)
- Diversified signal sources

## Technical Architecture

The system consists of several key components:

1. **OrderBook Class**: Maintains full limit order book state with realistic dynamics
2. **HFTSignalGenerator Class**: Calculates all trading signals and manages execution
3. **Signal Calculation Methods**: Individual methods for each microstructure indicator
4. **Trading Simulation Engine**: Executes trades based on composite signals
5. **Visualization System**: Creates comprehensive charts using Plotly
6. **Gradio Interface**: Provides interactive parameter control

## Performance Considerations

- **Computational Efficiency**: Optimized for high-frequency data processing
- **Memory Management**: Efficient storage of tick data and order book state
- **Scalability**: Can handle up to 1000 ticks/second simulation
- **Real-Time Visualization**: Updates charts dynamically without lag

## Educational Value

This project demonstrates several important concepts in quantitative finance:
- Market microstructure and order book dynamics
- Signal generation from order flow data
- Composite signal construction and weighting
- Risk management through position limits
- Performance measurement and attribution

## Limitations & Disclaimers

- This is a simplified simulation for educational purposes
- Does not include transaction costs, slippage, or market impact
- Assumes perfect execution at displayed prices
- Random order flow generation may not reflect all market conditions
- Not intended for actual trading without significant modifications

## Future Enhancements

Potential improvements include:
- Machine learning for dynamic weight optimization
- More sophisticated order flow modeling
- Transaction cost analysis
- Multi-asset correlation analysis
- Latency simulation and optimization
- Advanced risk management features

## Research Applications

This platform can be used for:
- Testing new signal generation methods
- Analyzing market microstructure patterns
- Optimizing signal weights and thresholds
- Studying order book dynamics
- Educational demonstrations of HFT concepts

## License

This project is licensed under the MIT License, allowing for both personal and commercial use with attribution.

## Author

Spencer Purdy