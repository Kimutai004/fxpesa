import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = 'LOE8UO0DV2CKIMR'

base_url = 'https://www.alphavantage.co/query'

params = {
    'function': 'FX_DAILY',
    'from_symbol': 'AUD',
    'to_symbol': 'USD',
    'apikey': API_KEY
}

def fetch_data():
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'Time Series FX (Daily)' in data:
            time_series = data['Time Series FX (Daily)']
            df = pd.DataFrame(time_series).T
            df.reset_index(inplace=True)
            df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']
            df['DATE'] = pd.to_datetime(df['DATE'])
            df.sort_values(by='DATE', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Convert columns to numeric
            df['OPEN'] = pd.to_numeric(df['OPEN'], errors='coerce')
            df['HIGH'] = pd.to_numeric(df['HIGH'], errors='coerce')
            df['LOW'] = pd.to_numeric(df['LOW'], errors='coerce')
            df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')

            return df
        else:
            raise ValueError('Error: Time Series FX (Daily) data not found in API response')
    else:
        raise ValueError('Failed to fetch data from Alpha Vantage API')

def head_and_shoulders(data, lookback=10):
    highest_high = data['HIGH'].rolling(window=lookback).max()
    lowest_low = data['LOW'].rolling(window=lookback).min()

    if len(data) < lookback:
        return 'Insufficient data for analysis'

    filtered_data = data[data['CLOSE'] > highest_high]
    if filtered_data.empty:
        return 'No head and shoulders pattern found'

    head = filtered_data.iloc[-1]
    left_shoulder = filtered_data.iloc[-2]
    right_shoulder = filtered_data.iloc[-3]

    conditions_sell = (
        left_shoulder['HIGH'] < head['HIGH'] > right_shoulder['HIGH']
        and left_shoulder['LOW'] < head['LOW'] > right_shoulder['LOW']
        and left_shoulder['CLOSE'] < right_shoulder['CLOSE']
    )

    conditions_buy = (
        left_shoulder['HIGH'] > head['HIGH'] < right_shoulder['HIGH']
        and left_shoulder['LOW'] > head['LOW'] < right_shoulder['LOW']
        and left_shoulder['CLOSE'] > right_shoulder['CLOSE']
    )

    return 'Bearish head and shoulders pattern detected' if conditions_sell else (
        'Bullish head and shoulders pattern detected' if conditions_buy else 'No head and shoulders pattern detected'
    )

def identify_patterns(data):
    double_bottoms = []
    double_tops = []

    for i in range(20, len(data)):
        conditions_bottom = (
            data['LOW'].iloc[i - 2] < data['LOW'].iloc[i - 1] < data['LOW'].iloc[i]
            and data['CLOSE'].iloc[i - 2] > data['CLOSE'].iloc[i - 1] > data['CLOSE'].iloc[i]
        )

        conditions_top = (
            data['HIGH'].iloc[i - 2] > data['HIGH'].iloc[i - 1] > data['HIGH'].iloc[i]
            and data['CLOSE'].iloc[i - 2] < data['CLOSE'].iloc[i - 1] < data['CLOSE'].iloc[i]
        )

        if conditions_bottom:
            double_bottoms.append(i)

        if conditions_top:
            double_tops.append(i)

    return double_bottoms, double_tops

def identify_wedge_patterns(data, window=10):
    rising_wedges = []
    falling_wedges = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    for i in range(1, len(data)):
        conditions_rising = (
            data['HIGH'].iloc[i] < highest_high.iloc[i]
            and data['LOW'].iloc[i] > lowest_low.iloc[i]
        )

        conditions_falling = (
            data['HIGH'].iloc[i] > highest_high.iloc[i]
            and data['LOW'].iloc[i] < lowest_low.iloc[i]
        )

        if conditions_rising:
            rising_wedges.append(i)

        if conditions_falling:
            falling_wedges.append(i)

    valid_patterns = [
        rising_wedges[i]
        for i in range(1, len(rising_wedges))
        if data['HIGH'].iloc[rising_wedges[i]] < data['HIGH'].iloc[rising_wedges[i - 1]]
        and data['LOW'].iloc[rising_wedges[i]] > data['LOW'].iloc[rising_wedges[i - 1]]
    ]

    valid_patterns += [
        falling_wedges[i]
        for i in range(1, len(falling_wedges))
        if data['HIGH'].iloc[falling_wedges[i]] > data['HIGH'].iloc[falling_wedges[i - 1]]
        and data['LOW'].iloc[falling_wedges[i]] < data['LOW'].iloc[falling_wedges[i - 1]]
    ]

    return valid_patterns

def identify_wedge_continuation_patterns(data, window=10):
    rising_wedges = []
    falling_wedges = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    for i in range(1, len(data)):
        conditions_rising = (
            data['HIGH'].iloc[i] < highest_high.iloc[i]
            and data['LOW'].iloc[i] > lowest_low.iloc[i]
            and data['HIGH'].iloc[i] > data['HIGH'].iloc[i - 1]
            and data['LOW'].iloc[i] < data['LOW'].iloc[i - 1]
        )

        conditions_falling = (
            data['HIGH'].iloc[i] > highest_high.iloc[i]
            and data['LOW'].iloc[i] < lowest_low.iloc[i]
            and data['HIGH'].iloc[i] < highest_high.iloc[i - 1]
            and data['LOW'].iloc[i] > lowest_low.iloc[i - 1]
        )

        if conditions_rising:
            rising_wedges.append(i)

        if conditions_falling:
            falling_wedges.append(i)

    valid_patterns = [
        rising_wedges[i]
        for i in range(1, len(rising_wedges))
        if data['HIGH'].iloc[rising_wedges[i]] > data['HIGH'].iloc[rising_wedges[i - 1]]
        and data['LOW'].iloc[rising_wedges[i]] < data['LOW'].iloc[rising_wedges[i - 1]]
    ]

    valid_patterns += [
        falling_wedges[i]
        for i in range(1, len(falling_wedges))
        if data['HIGH'].iloc[falling_wedges[i]] < data['HIGH'].iloc[falling_wedges[i - 1]]
        and data['LOW'].iloc[falling_wedges[i]] > data['LOW'].iloc[falling_wedges[i - 1]]
    ]

    return valid_patterns

def identify_flag_patterns(data, window=10):
    bull_flags = []
    bear_flags = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    for i in range(1, len(data)):
        conditions_bull = (
            data['HIGH'].iloc[i] < highest_high.iloc[i]
            and data['LOW'].iloc[i] > lowest_low.iloc[i]
            and data['LOW'].iloc[i] > data['LOW'].iloc[i - 1]
            and data['HIGH'].iloc[i] < data['HIGH'].iloc[i - 1]
            and (data['HIGH'].iloc[i] - lowest_low.iloc[i]) / (highest_high.iloc[i] - lowest_low.iloc[i]) < 0.5
        )

        conditions_bear = (
            data['HIGH'].iloc[i] > highest_high.iloc[i]
            and data['LOW'].iloc[i] < lowest_low.iloc[i]
            and data['HIGH'].iloc[i] < highest_high.iloc[i - 1]
            and data['LOW'].iloc[i] > lowest_low.iloc[i - 1]
            and (highest_high.iloc[i] - data['LOW'].iloc[i]) / (highest_high.iloc[i] - lowest_low.iloc[i]) < 0.5
        )

        if conditions_bull:
            bull_flags.append(i)

        if conditions_bear:
            bear_flags.append(i)

    valid_bull_flags = [
        bull_flags[i]
        for i in range(1, len(bull_flags))
        if data['HIGH'].iloc[bull_flags[i]] > data['HIGH'].iloc[bull_flags[i - 1]]
        and data['LOW'].iloc[bull_flags[i]] < data['LOW'].iloc[bull_flags[i - 1]]
    ]

    valid_bear_flags = [
        bear_flags[i]
        for i in range(1, len(bear_flags))
        if data['HIGH'].iloc[bear_flags[i]] < data['HIGH'].iloc[bear_flags[i - 1]]
        and data['LOW'].iloc[bear_flags[i]] > data['LOW'].iloc[bear_flags[i - 1]]
    ]

    return valid_bull_flags, valid_bear_flags

def identify_triangle_patterns(data, window=10):
    ascending_triangles = []
    descending_triangles = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    for i in range(1, len(data)):
        conditions_asc = (
            data['HIGH'].iloc[i] < highest_high.iloc[i]
            and data['LOW'].iloc[i] > lowest_low.iloc[i]
            and data['LOW'].iloc[i] > data['LOW'].iloc[i - 1]
            and data['HIGH'].iloc[i] > data['HIGH'].iloc[i - 1]
        )

        conditions_desc = (
            data['HIGH'].iloc[i] > highest_high.iloc[i]
            and data['LOW'].iloc[i] < lowest_low.iloc[i]
            and data['HIGH'].iloc[i] < data['HIGH'].iloc[i - 1]
            and data['LOW'].iloc[i] < data['LOW'].iloc[i - 1]
        )

        if conditions_asc:
            ascending_triangles.append(i)

        if conditions_desc:
            descending_triangles.append(i)

    return ascending_triangles, descending_triangles

def is_pin_bar(candle):
    body = abs(candle['OPEN'] - candle['CLOSE'])
    tail = candle['LOW'] - min(candle['OPEN'], candle['CLOSE'])
    head = max(candle['OPEN'], candle['CLOSE']) - candle['HIGH']

    is_bullish_pin = tail > 2 * body and head < body
    is_bearish_pin = head > 2 * body and tail < body

    return is_bullish_pin or is_bearish_pin

def identify_pin_bars(data):
    pin_bars = []

    for i in range(len(data)):
        if is_pin_bar(data.iloc[i]):
            pin_bars.append(i)

    return pin_bars

def is_bullish_engulfing(candle1, candle2):
    return candle1['OPEN'] > candle1['CLOSE'] and candle2['OPEN'] < candle2['CLOSE'] and candle2['OPEN'] < candle1['CLOSE'] and candle2['CLOSE'] > candle1['OPEN']

def is_bearish_engulfing(candle1, candle2):
    return candle1['OPEN'] < candle1['CLOSE'] and candle2['OPEN'] > candle2['CLOSE'] and candle2['OPEN'] > candle1['CLOSE'] and candle2['CLOSE'] < candle1['OPEN']

def identify_engulfing_candles(data):
    bullish_engulfing = []
    bearish_engulfing = []

    for i in range(1, len(data)):
        if is_bullish_engulfing(data.iloc[i - 1], data.iloc[i]):
            bullish_engulfing.append(i)
        elif is_bearish_engulfing(data.iloc[i - 1], data.iloc[i]):
            bearish_engulfing.append(i)

    return bullish_engulfing, bearish_engulfing

def predict_buy_sell(data):
    data['RETURN'] = data['CLOSE'].pct_change()
    data.dropna(inplace=True)

    X = data[['OPEN', 'HIGH', 'LOW', 'CLOSE']]
    y = (data['RETURN'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    data['PREDICTION'] = model.predict(X)

    return data


def generate_signals(patterns):
    signals = []

    for pattern in patterns:
        if pattern in ['Double Bottoms', 'Double Tops', 'Wedge Patterns', 'Wedge Continuation Patterns', 'Bull Flags', 'Bear Flags', 'Ascending Triangles', 'Descending Triangles']:
            signals.append('Sell' if patterns[pattern] else 'No Signal')
        else:
            signals.append('Buy' if patterns[pattern] else 'No Signal')

    return signals

def combine_signals(patterns, ml_predictions, data):
    signals = generate_signals(patterns)
    ml_signal = ml_predictions['PREDICTION'].values[-1]  # Get the latest ML prediction

    # Calculate the current price
    current_price = data['CLOSE'].iloc[-1]
    print("Current Price:", current_price)

    print("Signals:", signals)
    print("ML Prediction:", ml_signal)

    combined_signal = 'Hold'

    # Count the number of buy and sell signals
    num_buy_signals = signals.count('Buy')
    num_sell_signals = signals.count('Sell')

    # Determine the final signal based on pattern signals
    if num_sell_signals > num_buy_signals:
        combined_signal = 'Sell'
    elif num_buy_signals > num_sell_signals:
        combined_signal = 'Buy'
    else:  # If buy and sell signals are equal, use ML prediction
        combined_signal = 'Buy' if ml_signal == 1 else 'Sell'

    # Determine if a stronger buy or sell signal is present
    if combined_signal == 'Buy':
        if 'Buy' in signals and 'Sell' not in signals:
            if current_price > data['HIGH'].max():
                combined_signal = f'Buy Stop at {current_price:.2f}'
            else:
                combined_signal = f'Buy Limit at {current_price:.2f}'
    elif combined_signal == 'Sell':
        if 'Sell' in signals and 'Buy' not in signals:
            if current_price < data['LOW'].min():
                combined_signal = f'Sell Stop at {current_price:.2f}'
            else:
                combined_signal = f'Sell Limit at {current_price:.2f}'

    return combined_signal


def main():
    data = fetch_data()

    # Identify patterns
    double_bottoms, double_tops = identify_patterns(data)
    wedge_patterns = identify_wedge_patterns(data)
    wedge_continuation_patterns = identify_wedge_continuation_patterns(data)
    bull_flags, bear_flags = identify_flag_patterns(data)
    ascending_triangles, descending_triangles = identify_triangle_patterns(data)
    pin_bars = identify_pin_bars(data)
    bullish_engulfing, bearish_engulfing = identify_engulfing_candles(data)

    patterns = {
        'Double Bottoms': double_bottoms,
        'Double Tops': double_tops,
        'Wedge Patterns': wedge_patterns,
        'Wedge Continuation Patterns': wedge_continuation_patterns,
        'Bull Flags': bull_flags,
        'Bear Flags': bear_flags,
        'Ascending Triangles': ascending_triangles,
        'Descending Triangles': descending_triangles,
        'Pin Bars': pin_bars,
        'Bullish Engulfing': bullish_engulfing,
        'Bearish Engulfing': bearish_engulfing
    }

    # Predict buy/sell using machine learning
    ml_predictions = predict_buy_sell(data)

    # Combine signals
    final_signal = combine_signals(patterns, ml_predictions, data)
    print(f'Final Trading Signal: {final_signal}')

if __name__ == "__main__":
    main()
