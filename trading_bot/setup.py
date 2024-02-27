from setuptools import setup, find_packages

setup(
    name='trading_bot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'ccxt',    # For cryptocurrency exchange trading
        'pandas',  # For data manipulation
        'aiohttp', # For asynchronous HTTP requests
        'asyncio', # For asynchronous programming
        'pymongo', # For MongoDB interactions
        'talib',   # For technical analysis indicators (if you're using it)
        # Add any other dependencies your project needs
    ],
)