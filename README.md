# Portfolio Tracker

A command-line interface (CLI) application for tracking and simulating investment portfolios.

## Project Overview

This application allows users to manage an investment portfolio, track asset performance, and run sophisticated Monte Carlo simulations to assess future portfolio outcomes under risk and uncertainty.

## Features

### Core Features

1. **Add Assets** - Specify ticker, sector, asset class, quantity and purchase price.
2. **Price Tracking** - View current and historical prices with interactive graphs for individual or multiple tickers.
3. **Portfolio view** - Display all assets with name, sector, asset class, quantity, purchase price, transaction value, and current value.
4. **Portfolio Calculations** - Total portfolio value and relative weights of each asset, with breakdowns by asset class and sector.
5. **Simulation** - Monte Carlo simulation for a large number of sample paths over a long period demonstrating risk and uncertainty impact.

### Extended Features

- **Two Simulation Methods**:
    - Correlated Geometric Brownian Motion (GMB) by a Cholesky Decomposition.
    - Regime-Switching GBM with Hidden Markov Models.
- **Risk Metrics**: Value at Risk (VaR) and Conditional VaR (CVaR).
- **Transaction History**: Complete buy/sell log with realized P&L calculations.
- **Flexible Data Sources**: Toggle between cached data and live Yahoo Finance data.
- **Input Validation**: Ticker validation to prevent invalid entries.
- **Visual Analytics**: Normalized price comparisons.
- **Track Returns**: Unrealized and Realized P&L calculations for the portfolio.

## Technical Implementation

## MVC Architecture

The applications strictly follows the Model-View-Controller design pattern:

- **Model** (`Model/portfolio.py`)
    - Stores asset and transaction data using pandas DataFrames.
    - Implements all calculations: portfolio value, weights, P&L, returns.
    - Handles Monte Carlo simulations with correlation matrices.
    - Implements Hidden Markov Models for regime detection.
    - No user interaction or display logic

- **View** (`View/cli_view.py`)
    - Manages all user interface elements and menus.
    - Handles data visualization.
    - Formats output tables and messages.
    - Provides input validation helpers.
    - No business logic or calculations.

- **Controller** (`Controller/controller.py`)
    - Coordinates between Model and View.
    - Processes user comands and routes them appropriately.
    - Manages application state and flow.
    - Handles error management and validation.

### Project Structure

```
portfolio_tracker/
├── Model/
│   └── portfolio.py       
├── View/
│   └── cli_view.py          
├── Controller/
│   └── controller.py   
├── data_cache/             
│   ├── AAPL_10yhistory.csv
│   ├── MSFT_10yhistory.csv
│   └── ...
├── main.py            
├── download_and_cache.py
├── requirements.txt        
└── README.md           
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/jarnoklink/portfolio-tracker.git
   cd portfolio_tracker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

The application will start in cache-only mode.

## Usage Guide

### Main Menu

After starting the application, you will see:

```
==================================================
Portfolio Tracker
==================================================
1. Add asset
2. Show historical prices and plot
3. Show current portfolio
4. Sell asset
5. View transaction history
6. View portfolio breakdown by sector or asset class
7. Simulate portfolio
8. Toggle data source (online/cache)
9. Reset portfolio
0. Exit
==================================================
```

### Example Workflow
**1. Add Assets to Portfolio**
```
Choose option: 1
Ticker: AAPL
Sector: Technology
Asset Class: Equity
Quantity: 10
Purchase price: 300
```

**2. Run Simulation** (Option 7)
- Choose option: 7
- Choose simulation method (1 or 2)
- Enter years: 15 (default)
- Enter paths: 100000 (default)
- View results:
  - Current portfolio value
  - Median projected value (15 years)
  - 5th percentile (worst case)
  - 95th percentile (best case)
  - Median return (total and annualized)
  - VaR and CVaR risk metrics

### Available Cached Tickers

Since the yfinance package has a rate limit and overloads quickly, the application includes 10-year historical data for:
- **Technology**: AAPL, AMD, AVGO, MSFT, NVDA, ORCL, PLTR
- **Financial Services**: BAC, JPM, MA, V
- **Consumer Cyclicals**: AMZN, HD, TSLA
- **Communication Services**: GOOG, GOOGL, META, NFLX
- **Healthcare**: ABBV, JNJ, LLY
- **Consumer Defensive**: COST, WMT
- **Energy**: XOM

## Simulation Methods

### Method 1: Correlated Geometric Brownian Motion
- Uses historical mean returns and volatility.
- Incorporates correlation matrix between assets.
- Assumes log-normal distribution of returns.

### Method 2: Regime-Switching GBM
- Uses Hidden Markov Models to identify market regimes.
- Detects bull/bear market states from historical data.
- Transition between regimes based on probability matrix.
- Simulates each day with regime-specific parameters.

### Output Metrics

Both methods provide:
- **Current Value**: Starting portfolio value
- **Median Value**: Expected value after simulation period
- **5th Percentile**: Value exceeded in 95% of scenarios (downside risk)
- **95th Percentile**: Value exceeded in only 5% of scenarios (upside potential)
- **Median Return**: Total return and annualized compound return
- **VaR (5%)**: Maximum expected loss at 95% confidence level
- **CVaR (5%)**: Average loss in worst 5% of scenarios

## Dependencies

The project uses the following Python packages (specified in `requirements.txt`):

```
pandas>=2.2.3
numpy>=2.3.4
matplotlib>=3.10.0
yfinance>=0.2.57
hmmlearn>=0.3.3
scikit-learn>=1.6.1
```

Install all at once: 
```bash
pip install -r requirements.txt
```

## Data Source Management

### Cache-Only Mode (Default)
- Uses pre-downloaded historical data
- Fast and reliable
- No network dependency

### Online Mode
- Fetches live data from Yahoo Finance
- All tickers available
- May encouter rate limits
- Switch using Option 8 in menu

### Updating Cached Data

To download fresh data or add new tickers:

1. **Edit** `download_and_cache.py`:
    ```python
    TICKERS = ["AAPL", "MSFT", "YOUR_NEW_TICKER"]
    PERIOD = "10y" #Adjust as needed
    ```

2. **Run the download script**:
    ```bash
    python download_and_cache.py
    ```

3. **New data will be saved** to the `data_cache/` folder

This is useful when:
- You want to add new tickers.
- You need more recent data.
- You want a different time period (5y, 2y, etc.).

## Notes

- Cache-Only mode is recommended to avoid Yahoo Finance rate limits. For using other tickers, downloading historical data using `download_and_cache.py` is less likely to hit the rate limit.
- Simulations may take 20-60 seconds depending on settings.

## Author

Created as a demonstation of MVC architecture and financial portfolio simulation techniques.

Jarno Klink
GitHub: [@jarnoklink](https://github.com/jarnoklink)
