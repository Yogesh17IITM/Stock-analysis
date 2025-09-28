import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def find_best_days_to_trade(prices):
    """
    Find the best days to buy and sell stock to maximize profit.
    
    Args:
        prices (pd.Series): Series of daily stock prices
    
    Returns:
        tuple: (buy_date, sell_date, max_profit) where dates are datetime objects
    """
    if len(prices) < 2:
        return (None, None, 0)
    
    prices_list = prices.values  # Convert to numpy array for faster processing
    dates = prices.index
    
    min_price_day = 0
    current_min_day = 0
    max_profit = 0
    buy_day = 0
    sell_day = 0
    
    for current_day in range(1, len(prices_list)):
        if prices_list[current_day] < prices_list[current_min_day]:
            current_min_day = current_day
        
        current_profit = prices_list[current_day] - prices_list[current_min_day]
        
        if current_profit > max_profit:
            max_profit = current_profit
            buy_day = current_min_day
            sell_day = current_day
    
    return (dates[buy_day], dates[sell_day], max_profit)

# 🔍 Define the stock ticker (e.g., Reliance Industries on NSE)
ticker = "RELIANCE.NS"  # For Indian stocks, use .NS for NSE and .BO for BSE

# 📅 Define the time period (extending to 30 days for better analysis)
data = yf.download(ticker, period="30d", interval="1d")

# 📄 Display the data
print("\nStock Data:")
print(data)

# Find best trading days
buy_date, sell_date, max_profit = find_best_days_to_trade(data['Close'])

print("\n📊 Trading Analysis Results:")
if max_profit > 0:
    buy_price = float(data.loc[buy_date, 'Close'].iloc[0])  # Using iloc[0] to get scalar value
    sell_price = float(data.loc[sell_date, 'Close'].iloc[0])  # Using iloc[0] to get scalar value
    profit = float(max_profit.item())  # Properly convert numpy scalar to Python float
    print(f"Best day to buy: {buy_date.strftime('%Y-%m-%d')} at ₹{buy_price:.2f}")
    print(f"Best day to sell: {sell_date.strftime('%Y-%m-%d')} at ₹{sell_price:.2f}")
    print(f"Maximum profit per share: ₹{profit:.2f}")
else:
    print("No profitable trading opportunity found in this period")

# 📈 Plot the closing prices with buy/sell points
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], marker='o', linestyle='-', label='Closing Price')

if max_profit > 0:
    plt.plot(buy_date, data.loc[buy_date, 'Close'], 'g^', markersize=15, label='Buy Point')
    plt.plot(sell_date, data.loc[sell_date, 'Close'], 'rv', markersize=15, label='Sell Point')

plt.title(f"{ticker} - 30 Days Trading Analysis")
plt.xlabel("Date")
plt.ylabel("Closing Price (INR)")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
