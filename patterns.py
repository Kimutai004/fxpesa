import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

def fetch_data_from_api():
    # Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
    API_KEY = 'LOE8UO0DV2CKIMR7'

    # Define the base URL for the Alpha Vantage API
    base_url = 'https://www.alphavantage.co/query'

    # Define the parameters for the API request
    params = {
        'function': 'FX_DAILY',
        'from_symbol': 'EUR',
        'to_symbol': 'USD',
        'apikey': API_KEY
    }

    # Make the HTTP GET request to the Alpha Vantage API
    response = requests.get(base_url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Check if 'Time Series FX (Daily)' key exists in the response
        if 'Time Series FX (Daily)' in data:
            # Extract the time series data
            time_series = data['Time Series FX (Daily)']
            # Convert the time series data to a DataFrame
            df = pd.DataFrame(time_series).T
            # Reset index and rename columns
            df.reset_index(inplace=True)
            df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']
            # Convert date column to datetime format
            df['DATE'] = pd.to_datetime(df['DATE'])
            # Sort DataFrame by date
            df.sort_values(by='DATE', inplace=True)
            # Reset index
            df.reset_index(drop=True, inplace=True)
            # Print the DataFrame
            print(df)
        else:
            print('Error: Time Series FX (Daily) data not found in API response')
    else:
        print('Failed to fetch data from Alpha Vantage API')


def head_and_shoulders(data, lookback=20):
    if data.empty:
        return 'DataFrame is empty'

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
    if 'LOW' not in data.columns or 'CLOSE' not in data.columns:
        return [], []

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


def identify_wedge_patterns(data, window=20):
    rising_wedges = []
    falling_wedges = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    rising_wedges.append(0)
    falling_wedges.append(0)

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


def identify_wedge_continuation_patterns(data, window=20):
    rising_wedges = []
    falling_wedges = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    rising_wedges.append(0)
    falling_wedges.append(0)

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


def identify_flag_patterns(data, window=20):
    bull_flags = []
    bear_flags = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    bull_flags.append(0)
    bear_flags.append(0)

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


def identify_triangle_patterns(data, window=20):
    ascending_triangles = []
    descending_triangles = []

    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    ascending_triangles.append(0)
    descending_triangles.append(0)

    for i in range(1, len(data)):
        conditions_asc = (
            data['HIGH'].iloc[i] < highest_high.iloc[i]
            and data['LOW'].iloc[i] > lowest_low.iloc[i]
            and data['HIGH'].iloc[i] < data['HIGH'].iloc[i - 1]
            and data['LOW'].iloc[i] > data['LOW'].iloc[i - 1]
        )

        conditions_desc = (
            data['HIGH'].iloc[i] > highest_high.iloc[i]
            and data['LOW'].iloc[i] < lowest_low.iloc[i]
            and data['HIGH'].iloc[i] > data['HIGH'].iloc[i - 1]
            and data['LOW'].iloc[i] < data['LOW'].iloc[i - 1]
        )

        if conditions_asc:
            ascending_triangles.append(i)

        if conditions_desc:
            descending_triangles.append(i)

    valid_asc_triangles = [
        ascending_triangles[i]
        for i in range(1, len(ascending_triangles))
        if data['LOW'].iloc[ascending_triangles[i]] > data['LOW'].iloc[ascending_triangles[i - 1]]
    ]

    valid_desc_triangles = [
        descending_triangles[i]
        for i in range(1, len(descending_triangles))
        if data['HIGH'].iloc[descending_triangles[i]] < data['HIGH'].iloc[descending_triangles[i - 1]]
    ]

    return valid_asc_triangles, valid_desc_triangles


def is_pin_bar(candle):
    body_size = abs(candle['CLOSE'] - candle['OPEN'])
    wick_size = max(
        candle['HIGH'] - max(candle['CLOSE'], candle['OPEN']),
        max(candle['LOW'], min(candle['OPEN'], candle['CLOSE'])) - candle['LOW']
    )
    return wick_size >= 2 * body_size


def identify_pin_bars(data):
    pin_bars = []
    for i in range(1, len(data)):
        if is_pin_bar(data.iloc[i]):
            pin_bars.append(i)
    return pin_bars


def is_bullish_engulfing(candle1, candle2):
    return (
        candle2['OPEN'] < candle1['CLOSE']
        and candle2['CLOSE'] > candle1['OPEN']
        and candle2['HIGH'] > candle1['HIGH']
        and candle2['LOW'] < candle1['LOW']
    )


def is_bearish_engulfing(candle1, candle2):
    return (
        candle2['OPEN'] > candle1['CLOSE']
        and candle2['CLOSE'] < candle1['OPEN']
        and candle2['HIGH'] > candle1['HIGH']
        and candle2['LOW'] < candle1['LOW']
    )


def identify_engulfing_candles(data):
    engulfing_candles = []
    for i in range(2, len(data)):
        if is_bullish_engulfing(data.iloc[i - 1], data.iloc[i]):
            engulfing_candles.append((i - 1, i, 'Bullish Engulfing'))
        elif is_bearish_engulfing(data.iloc[i - 1], data.iloc[i]):
            engulfing_candles.append((i - 1, i, 'Bearish Engulfing'))
    return engulfing_candles


def preprocess_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, skiprows=1)
    data.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']
    return data


def predict_buy_sell(data):
    features = data[['HIGH', 'LOW', 'CLOSE']][:-1]
    data['Price_Up'] = data['CLOSE'].shift(-1) > data['CLOSE']
    target = data['Price_Up'][:-1]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')

    all_predictions = model.predict(features)

    buy_sell_predictions = ['Buy' if prediction else 'Sell' for prediction in all_predictions]

    return buy_sell_predictions


def generate_signals(patterns):
    signals = {}
    for pattern, direction in patterns.items():
        if direction == 'Sell':
            signals[pattern] = 'Sell Signal'
        elif direction == 'Buy':
            signals[pattern] = 'Buy Signal'
        else:
            signals[pattern] = 'No Signal'
    return signals

def combine_signals(patterns, ml_predictions):
    combined_signals = {}

    # Define priority order for signals
    signal_priority = [
        'Engulfing Candles',
        'Machine Learning',
        'Head and Shoulders',
        'Double Bottoms',
        'Double Tops',
        'Wedge Patterns',
        'Wedge Continuation Patterns',
        'Bull Flags',
        'Bear Flags',
        'Ascending Triangles',
        'Descending Triangles',
        'Pin Bars'
    ]

    # Initialize counters
    buy_count = 0
    sell_count = 0

    # Iterate through the priority order
    for signal_type in signal_priority:
        if signal_type == 'Machine Learning':
            # Add machine learning predictions
            for prediction in ml_predictions:
                if prediction == 'Buy':
                    buy_count += 1
                elif prediction == 'Sell':
                    sell_count += 1
        elif patterns[signal_type] in ['Buy', 'Sell']:
            # Add pattern-based signals
            if patterns[signal_type] == 'Buy':
                buy_count += 1
            elif patterns[signal_type] == 'Sell':
                sell_count += 1

    # Determine the final signal based on majority count
    final_signal = 'No Signal'
    if buy_count > sell_count:
        final_signal = 'Buy'
    elif sell_count > buy_count:
        final_signal = 'Sell'

    return final_signal


if __name__ == "__main__":
    data = pd.read_csv('C:/Users/User/Downloads/EURUSD_15.csv', sep='\t', header=None, skiprows=1)
    data.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']

    head_shoulders_pattern = head_and_shoulders(data)
    double_bottoms, double_tops = identify_patterns(data)
    wedge_patterns = identify_wedge_patterns(data)
    wedge_continuation_patterns = identify_wedge_continuation_patterns(data)
    bull_flags, bear_flags = identify_flag_patterns(data)
    asc_triangles, desc_triangles = identify_triangle_patterns(data)
    pin_bars = identify_pin_bars(data)
    engulfing_candles = identify_engulfing_candles(data)

    patterns = {
        'Head and Shoulders': head_shoulders_pattern,
        'Double Bottoms': 'Sell' if double_bottoms else 'No Signal',
        'Double Tops': 'Buy' if double_tops else 'No Signal',
        'Wedge Patterns': 'Sell' if wedge_patterns else 'No Signal',
        'Wedge Continuation Patterns': 'Buy' if wedge_continuation_patterns else 'No Signal',
        'Bull Flags': 'Sell' if bull_flags else 'No Signal',
        'Bear Flags': 'Sell' if bear_flags else 'No Signal',
        'Ascending Triangles': 'Sell' if asc_triangles else 'No Signal',
        'Descending Triangles': 'Sell' if desc_triangles else 'No Signal',
        'Pin Bars': 'Buy' if pin_bars else 'No Signal',
        'Engulfing Candles': 'Buy' if engulfing_candles else 'No Signal'
    }

    ml_predictions = predict_buy_sell(data)

    # Combine signals
    final_signal = combine_signals(patterns, ml_predictions)

    # Print the final signal
    print(f'Final Signal: {final_signal}')
