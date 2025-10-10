# Stock Analysis Configuration

# Stock ticker symbols (comma-separated)
# For Indian stocks: 
# - Use .NS suffix for NSE stocks (e.g., "RELIANCE.NS")
# - Use .BO suffix for BSE stocks (e.g., "TCS.BO")
# Example: "RELIANCE.NS, TCS.NS, INFY.NS"
STOCK_TICKERS = "RELIANCE.NS, TCS.NS, INFY.NS, BEL.NS, HDFCBANK.NS, SOLARINDS.NS, GRAPHITE.NS, ATHERENERG.NS, HINDCOPPER.NS, DRREDDY.NS, NATIONALUM.NS, INDIGO.NS, MMTC.NS"

# Analysis period in days (e.g., 30, 60, 90)
ANALYSIS_PERIOD = 7

# Data interval (1d for daily)
DATA_INTERVAL = "1d"

# Plot layout (number of plots to show in each row)
PLOTS_PER_ROW = 2