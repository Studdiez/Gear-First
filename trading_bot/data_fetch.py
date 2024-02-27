import logging
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)

free_crypto_api_key = 'i86kybehh7wk2sjawcoy'
free_crypto_base_url = 'https://api.freecryptoapi.com/v1'

headers = {'Authorization': f'Bearer {free_crypto_api_key}'}

# Semaphore to control the rate of requests
semaphore = asyncio.Semaphore(3)

async def fetch_symbols_data(client, symbol):
    endpoint = '/getData'
    params = {'symbol': symbol}

    url = f'{free_crypto_base_url}{endpoint}'

    retries = 3
    for attempt in range(retries):
        try:
            async with semaphore:
                async with client.get(url, params=params, headers=headers) as response:
                    data = await response.json()

                    # Print request URL and parameters for debugging
                    print(f'Request URL: {response.url}')
                    print(f'Request Parameters: {params}')
                    print(f'Response: {data}')

                    # Handle non-200 responses
                    if response.status != 200:
                        logging.error(f"Non-200 response from Free Crypto API: {response.status} - {data}")
                        return pd.DataFrame()  # Return an empty DataFrame

                    if 'symbols' in data and data['symbols']:
                        symbols_data_df = pd.DataFrame(data['symbols'])
                        # You can modify symbols_data_df or extract specific values if needed
                        return symbols_data_df
                    else:
                        logging.warning("No data available for the specified symbols.")
                        return pd.DataFrame()  # Return an empty DataFrame

        except Exception as e:
            logging.error(f"Error fetching symbols data (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                logging.info("Retrying...")
                await asyncio.sleep(5)
            else:
                logging.warning("Max retries reached. Skipping iteration.")
                return pd.DataFrame()  # Return an empty DataFrame

async def fetch_crypto_data_sync(symbol):
    async with aiohttp.ClientSession() as session:
        return await fetch_symbols_data(session, symbol)
    
    
    

async def main():
    # Example usage with dummy values
    symbol = 'XRPBTC@binance'

    while True:
        # Fetch data for XRP/BTC from Binance
        xrp_btc_data = await fetch_crypto_data_sync(symbol)

        # Print the result
        print("XRP/BTC Data:")
        print(xrp_btc_data)

        # Wait for 1/3 second before the next fetch
        await asyncio.sleep(1/3)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
