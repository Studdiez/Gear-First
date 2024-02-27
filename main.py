import os
import sys

# Dynamically add the trading_bot directory to sys.path
current_dir = os.path.dirname(__file__)
trading_bot_dir = r"C:\Users\tari1\Gear First\trading_bot"
if trading_bot_dir not in sys.path:
    sys.path.append(trading_bot_dir)

import asyncio
import importlib


# Perform dynamic imports
config_module = importlib.import_module("config")
data_fetch_module = importlib.import_module("data_fetch")
trading_strategy_module = importlib.import_module("trading_strategy")
execution_module = importlib.import_module("execution")

async def main():
    # Initialize the exchange client using credentials from config dynamically
    exchange = getattr(config_module, "ccxt").binance({
        'apiKey': getattr(config_module, "API_KEY"),
        'secret': getattr(config_module, "API_SECRET"),
        'enableRateLimit': True,
    })

async def main():
    
    # Initialize the exchange client using credentials from config dynamically
    exchange = getattr(config_module, "ccxt").binance({
        'apiKey': getattr(config_module, "API_KEY"),
        'secret': getattr(config_module, "API_SECRET"),
        'enableRateLimit': True,
    })

    while True:
        try:
            # Fetch market data
            market_data = await getattr(data_fetch_module, "fetch_market_data")()

            # Analyze market data to generate trading signals
            trading_signals = await getattr(trading_strategy_module, "analyze_data")(market_data)

            # Execute trades based on signals
            for signal in trading_signals:
                if signal['action'] == 'buy':
                    # Execute trade
                    await getattr(execution_module, "execute_trade")(exchange, signal['symbol'], signal['quantity'])
                    # Monitor and sell based on strategy
                    await getattr(execution_module, "monitor_and_sell")(exchange, signal['symbol'], signal['quantity'], signal['buy_price'])

            # The script was missing an `except` block for the `try` statement
        except Exception as e:
            print(f"An error occurred: {e}")

        # Wait before next cycle
        await asyncio.sleep(60)

if __name__ == '__main__':
    asyncio.run(main())
