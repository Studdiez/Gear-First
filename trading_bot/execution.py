import logging
import asyncio
import os
import ccxt
import numpy as np  # Add this line for NumPy
from binance.client import Client
from ccxt import NetworkError, ExchangeError
import ccxt.async_support as ccxt
from datetime import datetime, timedelta  # Add this line for datetime
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient


active_trades = {}    


ORDER_TYPE_MARKET = 'market'
SIDE_BUY = 'buy'
SIDE_SELL = 'sell'

api_key = os.getenv('EFxXUeSNt6zeezy3U3nBuUdJsK7qxaykEgOSmxGOXsMErawiR2PnsCFmZFGUqxsS')
api_secret = os.getenv('jSUr9WAlpCeX3UIL1MjHc6o31UMvAwbEG3xeoJWXNzMCPaGtkHGlcmHy9Y4pIYrJ')
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': api_key,
    'secret': api_secret,
    'timeout': 30000,
    'enableRateLimit': True,
})


# Ensure that the keys are available
if not api_key or not api_secret:
    raise ValueError("Binance API key and secret are required. Set them as environment variables.")

# Semaphore for rate limiting
max_requests_per_period = 50  # Binance rate limit: 50 requests per 10 seconds
rate_limit_semaphore = asyncio.Semaphore(max_requests_per_period)

async def limit_request(func, *args, **kwargs):
    async with rate_limit_semaphore:
        return await func(*args, **kwargs)
    
from pymongo import MongoClient

# Retrieve keys from environment variables
mongo_public_key = os.getenv('MONGO_PUBLIC_KEY')
mongo_private_key = os.getenv('MONGO_PRIVATE_KEY')

username = 'Studdiez'
password = 'Priceoffreedom'  # Make sure to URL encode your password if it contains special characters
cluster_url = 'cluster0.zloax10.mongodb.net'
dbname = 'Gear_One'  # Replace 'yourDatabaseName' with the name of your database

connection_string = f"mongodb+srv://{username}:{password}@{cluster_url}/{dbname}?retryWrites=true&w=majority"

client = MongoClient(connection_string)
db = client.trading_bot
trades_collection = db.trades


# Example: Test connection by listing database names
print(client.list_database_names())


# Proceed with database and collection selection
db = client['trading_bot']
active_trades_collection = db['active_trades']
historical_trades_collection = db['historical_trades']

async def fetch_current_price(symbol):
    """Fetch the current market price for a symbol."""
    ticker = await exchange.fetch_ticker(symbol)
    return ticker['last']

    
async def create_market_order(client, symbol, side, quantity):
    try:
        params = {
            'symbol': symbol,
            'side': side,
            'type': ORDER_TYPE_MARKET,
            'quantity': quantity
        }
        response = await execute_order(client, params)
        return response
    except Exception as e:
        logging.error(f"Error creating market order: {e}")
        return None


async def get_account_info(client):
    try:
        account_info = await client.fetch_balance()
        logging.info(f"Account Info: {account_info}")
        handle_errors(account_info)
    except Exception as e:
        logging.error(f"Error fetching account info: {e}")



def generate_signature(exchange, params):
    try:
        # Remove 'signature' if it's already present in params
        params.pop('signature', None)

        # Include the 'recvWindow' parameter in the ordered_params
        ordered_params = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Use the HMAC-SHA256 algorithm to create the signature
        signature = exchange.hmac(ordered_params.encode('utf-8'), api_secret.encode('utf-8'))

        # Inside generate_signature function
        logging.debug(f"Ordered Params: {ordered_params}")
        logging.debug(f"Generated Signature: {signature}")

        return signature
    except Exception as e:
        logging.error(f"Error generating signature: {e}")
        return None
    

def fetch_ohlcv_sync(client, symbol, interval, limit):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, client.fetch_ohlcv, symbol, interval, limit)



def create_trade_document(symbol, quantity, price, strategy):
    """Create a document for a new trade."""
    return {
        "symbol": symbol,
        "quantity": quantity,
        "entry_price": price,
        "strategy": strategy,
        "status": "active",
        "entry_time": datetime.utcnow()
    }

# Simplified Execute Order Function for Market Orders
async def execute_order(exchange, symbol, side, quantity):
    order_type = 'market'  # Explicitly defining order type as market
    order = await exchange.create_order(symbol, order_type, side, quantity)
    return order



async def execute_buy(symbol, quantity, strategy):
    # Execute buy market order
    order = await execute_order(exchange, symbol, 'buy', quantity)
    logging.info(f"Buy order executed: {order}")

    # Prepare trade document with details from the executed order
    trade_doc = {
        'symbol': symbol,
        'quantity': quantity,
        'entry_price': order['price'],  # Assuming 'price' is available in the order response
        'strategy': strategy,
        'status': 'active',
        'entry_time': datetime.utcnow()
    }
    # Insert trade document into MongoDB
    active_trades_collection.insert_one(trade_doc)
    logging.info("Trade document inserted into MongoDB.")


async def execute_sell(symbol, quantity, price):
    # Fetch the active trade document from MongoDB
    trade_doc = active_trades_collection.find_one({'symbol': symbol, 'status': 'active'})

    if trade_doc:
        # Execute sell order on the exchange
        order = await exchange.create_order(symbol, 'market', 'sell', quantity, price)
        logging.info(f"Sell order executed: {order}")

        # Update the trade document in MongoDB
        active_trades_collection.update_one(
            {'_id': trade_doc['_id']},
            {'$set': {'status': 'closed', 'exit_price': price, 'exit_time': datetime.utcnow()}}
        )
        logging.info("Trade document updated in MongoDB.")


# MongoDB Document Update After Selling
def update_trade_document(trade_id, exit_price):
    sell_timestamp = datetime.utcnow()
    update_result = active_trades_collection.update_one(
        {'_id': trade_id},
        {'$set': {
            'status': 'completed',
            'exit_price': exit_price,
            'sell_timestamp': sell_timestamp
        }}
    )
    if update_result.modified_count == 1:
        logging.info("Trade document updated successfully.")
    else:
        logging.error("Failed to update the trade document.")

async def monitor_and_sell(symbol, quantity, buy_price, investment):
    """Monitor the specified symbol and execute a sell order based on profit or stop loss conditions."""
    while True:
        current_price = await fetch_current_price(symbol)
        profit_target = buy_price * 1.02  # 2% profit target
        stop_loss_price = buy_price * 0.95  # 5% stop loss

        if current_price >= profit_target or current_price <= stop_loss_price:
            # Execute sell order with negative quantity indicating sell
            sell_order_result = await execute_sell(symbol=symbol, quantity=-quantity, order_type='sell')
            
            # Document the sell order in MongoDB
            trades_collection.insert_one(sell_order_result)
            
            # Update trade document after selling
            sell_timestamp = datetime.utcnow()
            update_result = active_trades_collection.update_one(
                {'_id': sell_order_result['_id']},
                {'$set': {
                    'status': 'completed',
                    'exit_price': sell_order_result['price'],
                    'sell_timestamp': sell_timestamp
                }}
            )
            if update_result.modified_count == 1:
                logging.info("Trade document updated successfully.")
            else:
                logging.error("Failed to update the trade document.")
            
            break  # Exit the loop once the order is executed

        await asyncio.sleep(5)  # Check every minute for price updates

def handle_errors(response):
    if 'code' in response and response['code'] != 200:
        logging.error(f"Error response from Binance API: {response}")
        # Implement additional error handling logic as needed       

async def receive_trade_signal(signal):
    print(f"Received trade signal: {signal}")
    # Process the signal, for example, execute a trade
    if signal['action'] == 'buy':
        # Execute the buy order using the signal's details
        # This is a simplified example. Replace it with actual logic to execute orders.
        print(f"Executing buy order for {signal['symbol']} with quantity {signal['quantity']}")


async def main_loop():
    while True:
        try:
            # Placeholder for retrieving trade signals
            # Ensure this part is adapted to your actual signal retrieval method
            trade_signals = await get_trade_signals()  # Adapt this to your actual implementation

            for signal in trade_signals:
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                if action == 'buy':
                    logging.info(f"Executing buy order for {symbol} with quantity {quantity}.")
                    buy_order_result = await execute_buy(symbol, quantity, 'buy')

                    if buy_order_result:
                        # Assuming buy_order_result contains the buy price and quantity bought
                        buy_price = buy_order_result['price']
                        quantity_bought = buy_order_result['quantity']
                        investment = buy_price * quantity_bought
                        
                        # Launch monitor_and_sell without awaiting, to not block the main loop
                        asyncio.create_task(monitor_and_sell(symbol, quantity_bought, buy_price, investment))
                        
                # Add handling for 'sell' signals if needed

        except Exception as e:
            logging.error(f"Error in main loop: {e}")

        await asyncio.sleep(60)  # Adjust based on your needs

# Placeholder for get_trade_signals function
async def get_trade_signals():
    # Implement based on your application's architecture
    return []

if __name__ == "__main__":
    asyncio.run(main_loop())

    
async def main():
    exchange = ccxt.binance({
        'apiKey': os.environ.get('EFxXUeSNt6zeezy3U3nBuUdJsK7qxaykEgOSmxGOXsMErawiR2PnsCFmZFGUqxsS'),
        'secret': os.environ.get('jSUr9WAlpCeX3UIL1MjHc6o31UMvAwbEG3xeoJWXNzMCPaGtkHGlcmHy9Y4pIYrJ'),
    })

    try:
        while True:
            await main_loop(exchange)
            await asyncio.sleep(1)  # Add a small delay to avoid high CPU usage
    except KeyboardInterrupt:
        print("Script terminated by user.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.create_task(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()