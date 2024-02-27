import asyncio
import logging
import ccxt.async_support as ccxt
import talib
import numpy as np
import pandas as pd
import aiohttp
from trading_bot.data_fetch import fetch_symbols_data

logging.basicConfig(level=logging.INFO)
api_key = 'your_api_key'
api_secret = 'your_api_secret'
trading_pair = 'BTC/XRP'
free_crypto_api_key = 'your_free_crypto_api_key'
free_crypto_base_url = 'https://api.freecryptoapi.com/v1'
headers = {'Authorization': f'Bearer {free_crypto_api_key}'}
semaphore = asyncio.Semaphore(3)


def calculate_rsi(data, window=14):
    # Calculate Relative Strength Index (RSI)
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def identify_candlestick_patterns(data):
    """Identify various candlestick patterns."""
    open_prices = data['open'].astype(float)
    high_prices = data['high'].astype(float)
    low_prices = data['low'].astype(float)
    close_prices = data['close'].astype(float)
    
    patterns = {
        'doji': talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices),
        'bullish_engulfing': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
        'hammer': talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
        'morning_star': talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices),
        'hanging_man': talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices),
        'bull_elephant_bar': talib.CDLCONCEALEDBABYSWALL(open_prices, high_prices, low_prices, close_prices),
        'tweezer_bottom': talib.CDLTWEEZERBOTTOM(open_prices, high_prices, low_prices, close_prices)
    }
    
    # Return a dictionary with patterns as keys and boolean arrays as values
    return {pattern: values == 100 for pattern, values in patterns.items()}

def calculate_trade_score(rsi_last_value, ma20, ma200, patterns):
    """Integrate quant_score directly in the trade score calculation."""
    rsi_score = quant_score(rsi_last_value)
    score = 0.5 * rsi_score + 0.25 * quant_score(ma20) + 0.25 * quant_score(ma200)
    pattern_scores = {
        'doji': 0.1, 'bullish_engulfing': 0.2, 'hammer': 0.3, 'morning_star': 0.4,
        'hanging_man': -0.3, 'bull_elephant_bar': 0.5, 'tweezer_bottom': 0.5
    }
    for pattern, value in patterns.items():
        if value.any():
            score += pattern_scores.get(pattern, 0)
    return score

def quant_score(value):
    """Quantize indicator values into a score."""
    if value > 70:
        return 1.0
    elif value > 30:
        return 0.5
    else:
        return 0.0
    
some_threshold = 0.10  # Example threshold, adjust based on your strategy
portfolio_allocation_percent = 0.10  # Example portfolio allocation

async def generate_and_send_trade_signal():
    execution_module = __import__('execution')
    
    # Generate the trade signal
    current_balance = await execution_module.fetch_account_balance()
    trade_quantity = 0.1 * current_balance  # Calculate 10% of the current balance
    
    trade_signal = {
        "symbol": "BTC/XRP",
        "action": "buy",
        "quantity": trade_quantity
    }
    
    # Send the trade signal to execution.py
    await execution_module.receive_trade_signal(trade_signal)

    
async def trading_strategy():
    execution_module = __import__('execution')
    data_fetch_module = __import__('data_fetch.py')

            # Dynamically call fetch_current_price and fetch_account_balance
    current_balance = await execution_module.get_account_info()
    current_price = await data_fetch_module.fetch_symbols_data('BTC/XRP')
  
   
    
    # Ensure the exchange is correctly initialized in execution.py and accessible here
    exchange = execution_module.exchange 

    while True:
        trades = []
        symbol_data = await fetch_symbols_data('BTC/XRP')
        if symbol_data.empty:
            logging.warning("Data fetch returned empty.")
        else:
            symbol_data[['open', 'high', 'low', 'close']] = symbol_data[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
            patterns = identify_candlestick_patterns(symbol_data)
            rsi_last_value = calculate_rsi(symbol_data['close']).iloc[-1]
            ma20 = talib.SMA(symbol_data['close'], timeperiod=20).iloc[-1]
            ma200 = talib.SMA(symbol_data['close'], timeperiod=200).iloc[-1]
            score = calculate_trade_score(rsi_last_value, ma20, ma200, patterns)
            trades.append({'symbol': trading_pair, 'score': score})
            logging.info(f"Trade score for {'BTC/XRP'}: {score}")

            if trades:
                highest_scored_trade = max(trades, key=lambda x: x['score'])
                if highest_scored_trade['score'] > some_threshold:
                    # Fetch current price and account balance dynamically within the loop
                    current_price = await data_fetch_module.fetch_symbols_data('BTC/XRP')
                    current_balance = await execution_module.get_account_info()
                    calculated_quantity = (current_balance * portfolio_allocation_percent) / current_price
                    await generate_and_send_trade_signal(highest_scored_trade['BTC/XRP'], calculated_quantity)

        await asyncio.sleep(60)  # Sleep before the next iteration

if __name__ == "__main__":
    asyncio.run(trading_strategy())