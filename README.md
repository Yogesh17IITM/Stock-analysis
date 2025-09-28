# Stock Analysis Tool

This Python script helps you analyze stock data to find the optimal days for buying and selling stocks to maximize profit. The tool currently focuses on analyzing Reliance Industries stock (RELIANCE.NS) but can be modified for any stock listed on NSE or BSE.

## Features

- Fetches real-time stock data using yfinance
- Analyzes historical price data to find the best buying and selling opportunities
- Visualizes stock price movements with highlighted buy/sell points
- Calculates maximum potential profit per share
- Displays comprehensive stock data including Open, High, Low, Close prices, and Volume

## Requirements

- Python 3.x
- Required Python packages:
  - yfinance
  - pandas
  - matplotlib

## Installation

```bash
pip install yfinance pandas matplotlib
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Yogesh17IITM/Stock-analysis.git
```

2. Navigate to the project directory:
```bash
cd Stock-analysis
```

3. Run the script:
```bash
python test.py
```

The script will:
- Download the last 30 days of stock data
- Display a detailed price table
- Show the best days to buy and sell
- Calculate potential profit
- Display a graph with price movement and buy/sell points

## Example Output

The script provides:
- Comprehensive stock data table with daily prices
- Trading analysis results showing:
  - Best day to buy with price
  - Best day to sell with price
  - Maximum potential profit per share
- Interactive graph showing price movements with buy/sell points marked

## Customization

To analyze a different stock:
1. Open `test.py`
2. Modify the `ticker` variable:
   - For NSE stocks: Use `.NS` suffix (e.g., "TCS.NS")
   - For BSE stocks: Use `.BO` suffix (e.g., "TCS.BO")

## Contributing

Feel free to fork this repository and submit pull requests. You can also open issues for bugs or feature requests.

## License

This project is open source and available for anyone to use and modify.