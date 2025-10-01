import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import math
import numpy as np
import os

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

def read_config():
    """Read and parse the configuration file"""
    config = {}
    with open('config.rb', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = [x.strip() for x in line.split('=', 1)]
                # Remove quotes if present
                value = value.strip('"\'')
                config[key] = value
    return config

def plot_stock_data(data, ticker, ax, result):
    """
    Plot stock data and analysis results on the given axes
    
    Args:
        data (pd.DataFrame): Stock price data
        ticker (str): Stock ticker symbol
        ax (matplotlib.axes.Axes): Matplotlib axes object for plotting
        result (dict): Analysis results including buy/sell points and profit
    """
    # Plot basic price data with improved styling
    ax.plot(data.index, data['Close'], marker='o', markersize=4, 
            linestyle='-', linewidth=1.5, label='Closing Price', 
            color='#2E86C1')
    
    if result['profit'] > 0:
        # Plot buy and sell points with enhanced visibility
        ax.plot(result['buy_date'], result['buy_price'], 'g^', markersize=15, label='Buy Point',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
        ax.plot(result['sell_date'], result['sell_price'], 'rv', markersize=15, label='Sell Point',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
        
        # Add annotations with improved styling
        ax.annotate(f'Buy: Rs.{result["buy_price"]:.2f}', 
                   xy=(result['buy_date'], result['buy_price']),
                   xytext=(10, -20),
                   textcoords='offset points',
                   ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', fc='#A2D9CE', ec='green', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'))
        
        ax.annotate(f'Sell: Rs.{result["sell_price"]:.2f}', 
                   xy=(result['sell_date'], result['sell_price']),
                   xytext=(10, 20),
                   textcoords='offset points',
                   ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', fc='#F5B7B1', ec='red', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
        
        # Add profit annotation with improved visibility
        mid_date = result['buy_date'] + (result['sell_date'] - result['buy_date'])/2
        mid_price = (result['buy_price'] + result['sell_price'])/2
        ax.annotate(f'Profit: Rs.{result["profit"]:.2f}\n(+{result["profit_percentage"]:.1f}%)', 
                   xy=(mid_date, mid_price),
                   xytext=(0, 30),
                   textcoords='offset points',
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', fc='#D4E6F1', ec='blue', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='blue'))
    
    # Enhanced plot styling
    ax.set_title(f"{ticker} Trading Analysis", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Date", fontsize=10, labelpad=10)
    ax.set_ylabel("Price (INR)", fontsize=10, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    
    # Add legend with improved placement
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.1), 
             ncol=3, frameon=True, fancybox=True, shadow=True)

def analyze_and_plot_stock(data, ticker, ax):
    """
    Analyze and plot trading opportunities for a single stock
    
    Args:
        data (pd.DataFrame): Stock price data
        ticker (str): Stock ticker symbol
        ax (matplotlib.axes.Axes): Matplotlib axes object for plotting
    
    Returns:
        dict: Analysis results including buy/sell dates, prices, and profit
    """
    # Find best trading days
    buy_date, sell_date, max_profit = find_best_days_to_trade(data['Close'])
    
    result = {
        'ticker': ticker,
        'buy_date': None,
        'sell_date': None,
        'buy_price': 0,
        'sell_price': 0,
        'profit': 0,
        'profit_percentage': 0
    }
    
    if max_profit > 0:
        buy_price = float(data.loc[buy_date, 'Close'].iloc[0])
        sell_price = float(data.loc[sell_date, 'Close'].iloc[0])
        profit = float(max_profit.item())
        profit_percentage = (profit/buy_price*100)
        
        # Update result dictionary
        result.update({
            'buy_date': buy_date,
            'sell_date': sell_date,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'profit': profit,
            'profit_percentage': profit_percentage
        })
    
    # Plot the stock data on the provided axes
    plot_stock_data(data, ticker, ax, result)
    
    # Create directory if it doesn't exist
    os.makedirs('samples/individual_plots', exist_ok=True)
    
    # Save individual stock plot with proper figure management
    individual_fig = plt.figure(figsize=(10, 6))
    individual_ax = individual_fig.add_subplot(111)
    
    # Plot on the individual figure without recursive call
    plot_stock_data(data, ticker, individual_ax, result)
    
    individual_fig.tight_layout()
    
    # Save individual plot with high DPI
    individual_fig.savefig(f'samples/individual_plots/{ticker.replace(".", "_")}.png', 
                          dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(individual_fig)  # Close individual plot immediately
    
    # Write analysis results to text file with UTF-8 encoding
    with open(f'samples/individual_plots/{ticker.replace(".", "_")}_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(f"Analysis Results for {ticker}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n\n")
        
        if max_profit > 0:
            f.write(f"Best day to buy: {buy_date.strftime('%Y-%m-%d')} at Rs.{buy_price:.2f}\n")
            f.write(f"Best day to sell: {sell_date.strftime('%Y-%m-%d')} at Rs.{sell_price:.2f}\n")
            f.write(f"Maximum profit per share: Rs.{profit:.2f} ({profit_percentage:.1f}%)\n")
        else:
            f.write("No profitable trading opportunity found in this period\n")
        
        # Calculate statistics
        avg_price = float(data['Close'].mean())
        min_price = float(data['Close'].min())
        max_price = float(data['Close'].max())
        volatility = float(data['Close'].std())
        
        f.write("\nPrice Statistics:\n")
        f.write(f"Average Price: Rs.{avg_price:.2f}\n")
        f.write(f"Minimum Price: Rs.{min_price:.2f}\n")
        f.write(f"Maximum Price: Rs.{max_price:.2f}\n")
        f.write(f"Price Volatility: {volatility:.2f}\n")
    
    return result

def main():
    # Get configuration
    config = read_config()
    tickers = [t.strip() for t in config['STOCK_TICKERS'].split(',')]
    period = f"{config['ANALYSIS_PERIOD']}d"
    interval = config['DATA_INTERVAL']
    plots_per_row = int(config['PLOTS_PER_ROW'])
    
    # Calculate subplot layout
    num_stocks = len(tickers)
    num_rows = math.ceil(num_stocks / plots_per_row)
    num_cols = min(plots_per_row, num_stocks)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12 * num_cols / 2, 6 * num_rows))
    if num_stocks == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Store analysis results
    results = []
    
    # Analyze each stock
    print(f"\n📊 Trading Analysis Results for {num_stocks} stocks:")
    print(f"Analysis period: {config['ANALYSIS_PERIOD']} days")
    print("-" * 50)
    
    for idx, ticker in enumerate(tickers):
        print(f"\nAnalyzing {ticker}...")
        
        # Download stock data with explicit auto_adjust parameter
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        # Analyze and plot
        result = analyze_and_plot_stock(data, ticker, axes[idx])
        results.append(result)
        
        # Print results
        if result['profit'] > 0:
            print(f"Best day to buy: {result['buy_date'].strftime('%Y-%m-%d')} at Rs.{result['buy_price']:.2f}")
            print(f"Best day to sell: {result['sell_date'].strftime('%Y-%m-%d')} at Rs.{result['sell_price']:.2f}")
            print(f"Maximum profit per share: Rs.{result['profit']:.2f} ({result['profit_percentage']:.1f}%)")
        else:
            print("No profitable trading opportunity found in this period")
    
    # Hide empty subplots if any
    for idx in range(len(tickers), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n📈 Summary of Best Opportunities:")
    print("-" * 50)
    # Sort results by profit percentage
    sorted_results = sorted(results, key=lambda x: x['profit_percentage'], reverse=True)
    for result in sorted_results:
        if result['profit'] > 0:
            print(f"{result['ticker']}: Potential profit Rs.{result['profit']:.2f} ({result['profit_percentage']:.1f}%)")

if __name__ == "__main__":
    main()