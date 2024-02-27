# m1_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import numpy as np
from trading_bot.data_fetch import fetch_ohlcv_sync
from m1_model import calculate_trade_score_candlestick

def train_ml_model(features_and_labels):
    # Assume features_and_labels is a DataFrame with features and labels
    # Replace the following line with your actual data processing and model training logic
    X_train, X_test, y_train, y_test = train_test_split(features_and_labels.drop('label', axis=1),
                                                        features_and_labels['label'],
                                                        test_size=0.2, random_state=42)

    # Create and train the machine learning model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    return model

def predict_with_ml_model(model, current_features):
    # Assume current_features is a DataFrame with the current features for prediction
    # Replace the following line with your actual data processing and prediction logic
    prediction = model.predict(current_features)
    
    return prediction

def log_trade(executed_trade, log_file='trade_log.csv'):
    # Log the details of the executed trade into a CSV file
    try:
        trade_log = pd.read_csv(log_file)
    except FileNotFoundError:
        trade_log = pd.DataFrame(columns=['Timestamp', 'Symbol', 'Action', 'Result'])
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    trade_log = trade_log.append({'Timestamp': timestamp,
                                  'Symbol': executed_trade['symbol'],
                                  'Action': executed_trade['action'],
                                  'Result': executed_trade['result']}, ignore_index=True)
    
    trade_log.to_csv(log_file, index=False)

    print(f"Trade logged: {executed_trade}")

def load_historical_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    except FileNotFoundError:
        print(f"File not found at {file_path}. Returning an empty DataFrame.")
        return pd.DataFrame()

def save_historical_data(file_path, historical_data):
    historical_data.to_csv(file_path)

def update_historical_data(file_path, symbol, interval, limit):
    try:
        # Fetch new data
        new_data = fetch_ohlcv_sync(symbol, interval, limit)
        
        if not new_data.empty:
            # Load existing historical data
            historical_data = load_historical_data(file_path)
            
            # Append the new data
            historical_data = historical_data.append(new_data)
            
            # Save the updated historical data
            save_historical_data(file_path, historical_data)
            
            return historical_data
        else:
            print("No new data fetched. Historical data remains unchanged.")
            return load_historical_data(file_path)
    except Exception as e:
        print(f"Error updating historical data: {e}")
        return load_historical_data(file_path)

def train_classification_model(historical_data):
    # Feature engineering
    historical_data['ma20'] = historical_data['close'].rolling(window=20).mean()
    historical_data['ma200'] = historical_data['close'].rolling(window=200).mean()
    historical_data['rsi'] = calculate_rsi(historical_data['close'], window=14)
    
    # Labeling - 1 for buy, 0 for hold/sell
    historical_data['label'] = np.where(historical_data['close'].shift(-1) > historical_data['close'], 1, 0)
    
    # Drop NaN values introduced by rolling mean and shift
    historical_data.dropna(inplace=True)
    
    # Select features and labels
    X = historical_data[['ma20', 'ma200', 'rsi']]
    y = historical_data['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    
    return classifier

def calculate_trade_score(indicators, doji, hammer, morning_star):
    # Extract indicators
    ma20, ma200, rsi = indicators

    # Weights for each indicator
    weight_ma20 = 0.4
    weight_ma200 = 0.4
    weight_rsi = 0.2

    # Score calculation
    score = (
        weight_ma20 * quant_score(ma20) +
        weight_ma200 * quant_score(ma200) +
        weight_rsi * quant_score(rsi)
    )

def calculate_trade_score_candlestick(indicators, doji, hammer, morning_star, bullish_engulfing, hanging_man):
    
    if doji:
        score += 0.1
    if bullish_engulfing:
        score += 0.2
    if hammer:
        score += 0.3
    if hanging_man:
        score += 0.3
    if morning_star:
        score += 0.4
def quant_score(value):
    # Simple quantization logic
    if value > 70:
        return 1.0
    elif 30 < value <= 70:
        return 0.5
    else:
        return 0.0

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

def adapt_and_trade(strategy, symbol, interval, limit, file_path):
    # Update historical data
    historical_data = update_historical_data(file_path, symbol, interval, limit)
    
    # Train the classification model
    classifier = train_classification_model(historical_data)
    
    # Apply the strategy using the trained model
    strategy(historical_data, classifier)

def identify_candlestick_patterns(candlestick_data):
    # Implement candlestick identification logic
    open_prices = candlestick_data['open']
    close_prices = candlestick_data['close']
    high_prices = candlestick_data['high']
    low_prices = candlestick_data['low']

    # Identify Bullish Engulfing pattern
    bullish_engulfing = np.zeros(len(close_prices))
    for i in range(1, len(close_prices)):
        if close_prices[i - 1] < open_prices[i - 1] and close_prices[i] > open_prices[i] and \
           close_prices[i] > open_prices[i - 1] and open_prices[i] < close_prices[i - 1]:
            bullish_engulfing[i] = 1

    # Identify Hammer pattern
    hammer = np.zeros(len(close_prices))
    for i in range(len(close_prices)):
        if close_prices[i] < open_prices[i] and (high_prices[i] - close_prices[i]) < (open_prices[i] - close_prices[i]):
            hammer[i] = 1

    # Identify Morning Star pattern
    morning_star = np.zeros(len(close_prices))
    for i in range(2, len(close_prices)):
        if close_prices[i - 2] > open_prices[i - 2] and close_prices[i - 1] < open_prices[i - 1] and \
           close_prices[i] > open_prices[i] and close_prices[i] > open_prices[i - 2] and open_prices[i] < close_prices[i - 2]:
            morning_star[i] = 1

    # Identify Doji pattern
    doji = np.zeros(len(close_prices))
    for i in range(len(close_prices)):
        if abs(close_prices[i] - open_prices[i]) <= 0.01 * (high_prices[i] - low_prices[i]):
            doji[i] = 1

    # Identify Hanging Man pattern
    hanging_man = np.zeros(len(close_prices))
    for i in range(len(close_prices)):
        if close_prices[i] < open_prices[i] and (low_prices[i] - close_prices[i]) < (open_prices[i] - close_prices[i]):
            hanging_man[i] = 1

    return bullish_engulfing, hammer, morning_star, doji, hanging_man
    
