import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def head_and_shoulders(data, lookback=20):
    if data.empty:
        return 'DataFrame is empty'

    # Calculate the highest high and lowest low over the lookback period
    highest_high = data['HIGH'].rolling(window=lookback).max()
    lowest_low = data['LOW'].rolling(window=lookback).min()

    # Check if the DataFrame has enough elements
    if len(data) < lookback:
        return 'Insufficient data for analysis'

    # Identify the head and shoulders
    filtered_data = data[data['CLOSE'] > highest_high]
    if filtered_data.empty:
        return 'No head and shoulders pattern found'

    head = filtered_data.iloc[-1]

    # Check if the head is at the center of two shoulders
    left_shoulder = filtered_data.iloc[-2]
    right_shoulder = filtered_data.iloc[-3]

    # Conditions for head and shoulders pattern
    if left_shoulder['HIGH'] < head['HIGH'] > right_shoulder['HIGH'] \
            and left_shoulder['LOW'] < head['LOW'] > right_shoulder['LOW'] \
            and left_shoulder['CLOSE'] < right_shoulder['CLOSE']:
        return 'Bearish head and shoulders pattern detected'  # Price likely to move downwards
    elif left_shoulder['HIGH'] > head['HIGH'] < right_shoulder['HIGH'] \
            and left_shoulder['LOW'] > head['LOW'] < right_shoulder['LOW'] \
            and left_shoulder['CLOSE'] > right_shoulder['CLOSE']:
        return 'Bullish head and shoulders pattern detected'  # Price likely to move upwards
    else:
        return 'No head and shoulders pattern detected'

def identify_patterns(data):
    if 'LOW' not in data.columns or 'CLOSE' not in data.columns:
        return [], []  # Return empty lists if necessary columns are not present

    double_bottoms = []
    double_tops = []

    # Identify double bottoms
    for i in range(20, len(data)):
        if (data['LOW'].iloc[i - 2] < data['LOW'].iloc[i - 1] < data['LOW'].iloc[i] and
                data['CLOSE'].iloc[i - 2] > data['CLOSE'].iloc[i - 1] > data['CLOSE'].iloc[i]):
            double_bottoms.append(i)

    # Identify double tops
    for i in range(20, len(data)):
        if (data['HIGH'].iloc[i - 2] > data['HIGH'].iloc[i - 1] > data['HIGH'].iloc[i] and
                data['CLOSE'].iloc[i - 2] < data['CLOSE'].iloc[i - 1] < data['CLOSE'].iloc[i]):
            double_tops.append(i)

    return double_bottoms, double_tops

def identify_wedge_patterns(data, window=20):
    rising_wedges = []
    falling_wedges = []

    # Calculate the highest high and lowest low over the window period
    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    # Identify the rising wedge
    rising_wedges.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] < highest_high.iloc[i] and
                data['LOW'].iloc[i] > lowest_low.iloc[i]):
            rising_wedges.append(i)

    # Identify the falling wedge
    falling_wedges.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] > highest_high.iloc[i] and
                data['LOW'].iloc[i] < lowest_low.iloc[i]):
            falling_wedges.append(i)

    # Check if the patterns are valid
    valid_patterns = []
    for i in range(1, len(rising_wedges)):
        if (data['HIGH'].iloc[rising_wedges[i]] < data['HIGH'].iloc[rising_wedges[i - 1]] and
                data['LOW'].iloc[rising_wedges[i]] > data['LOW'].iloc[rising_wedges[i - 1]]):
            valid_patterns.append(rising_wedges[i])

    for i in range(1, len(falling_wedges)):
        if (data['HIGH'].iloc[falling_wedges[i]] > data['HIGH'].iloc[falling_wedges[i - 1]] and
                data['LOW'].iloc[falling_wedges[i]] < data['LOW'].iloc[falling_wedges[i - 1]]):
            valid_patterns.append(falling_wedges[i])

    return valid_patterns

def identify_wedge_continuation_patterns(data, window=20):
    rising_wedges = []
    falling_wedges = []

    # Calculate the highest high and lowest low over the window period
    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    # Identify the rising wedge
    rising_wedges.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] < highest_high.iloc[i] and
                data['LOW'].iloc[i] > lowest_low.iloc[i] and
                data['HIGH'].iloc[i] > data['HIGH'].iloc[i - 1] and
                data['LOW'].iloc[i] < data['LOW'].iloc[i - 1]):
            rising_wedges.append(i)

    # Identify the falling wedge
    falling_wedges.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] > highest_high.iloc[i] and
                data['LOW'].iloc[i] < lowest_low.iloc[i] and
                data['HIGH'].iloc[i] < highest_high.iloc[i - 1] and
                data['LOW'].iloc[i] > lowest_low.iloc[i - 1]):
            falling_wedges.append(i)

    # Check if the patterns are valid
    valid_patterns = []
    for i in range(1, len(rising_wedges)):
        if (data['HIGH'].iloc[rising_wedges[i]] > data['HIGH'].iloc[rising_wedges[i - 1]] and
                data['LOW'].iloc[rising_wedges[i]] < data['LOW'].iloc[rising_wedges[i - 1]]):
            valid_patterns.append(rising_wedges[i])

    for i in range(1, len(falling_wedges)):
        if (data['HIGH'].iloc[falling_wedges[i]] < data['HIGH'].iloc[falling_wedges[i - 1]] and
                data['LOW'].iloc[falling_wedges[i]] > data['LOW'].iloc[falling_wedges[i - 1]]):
            valid_patterns.append(falling_wedges[i])

    return valid_patterns

def identify_flag_patterns(data, window=20):
    bull_flags = []
    bear_flags = []

    # Calculate the highest high and lowest low over the window period
    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    # Identify the bull flag
    bull_flags.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] < highest_high.iloc[i] and
                data['LOW'].iloc[i] > lowest_low.iloc[i] and
                data['LOW'].iloc[i] > data['LOW'].iloc[i - 1] and
                data['HIGH'].iloc[i] < data['HIGH'].iloc[i - 1] and
                (data['HIGH'].iloc[i] - lowest_low.iloc[i]) / (highest_high.iloc[i] - lowest_low.iloc[i]) < 0.5):
            bull_flags.append(i)

    # Identify the bear flag
    bear_flags.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] > highest_high.iloc[i] and
                data['LOW'].iloc[i] < lowest_low.iloc[i] and
                data['HIGH'].iloc[i] < highest_high.iloc[i - 1] and
                data['LOW'].iloc[i] > lowest_low.iloc[i - 1] and
                (highest_high.iloc[i] - data['LOW'].iloc[i]) / (highest_high.iloc[i] - lowest_low.iloc[i]) < 0.5):
            bear_flags.append(i)

    # Check if the patterns are valid
    valid_patterns = []
    for i in range(1, len(bull_flags)):
        if (data['HIGH'].iloc[bull_flags[i]] > data['HIGH'].iloc[bull_flags[i - 1]] and
                data['LOW'].iloc[bull_flags[i]] < data['LOW'].iloc[bull_flags[i - 1]]):
            valid_patterns.append(bull_flags[i])

    for i in range(1, len(bear_flags)):
        if (data['HIGH'].iloc[bear_flags[i]] < data['HIGH'].iloc[bear_flags[i - 1]] and
                data['LOW'].iloc[bear_flags[i]] > data['LOW'].iloc[bear_flags[i - 1]]):
            valid_patterns.append(bear_flags[i])

    return valid_patterns

def identify_triangle_patterns(data, window=20):
    ascending_triangles = []
    descending_triangles = []

    # Calculate the highest high and lowest low over the window period
    highest_high = data['HIGH'].rolling(window=window).max()
    lowest_low = data['LOW'].rolling(window=window).min()

    # Identify the ascending triangle
    ascending_triangles.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] < highest_high.iloc[i] and
                data['LOW'].iloc[i] > lowest_low.iloc[i] and
                data['HIGH'].iloc[i] < data['HIGH'].iloc[i - 1] and
                data['LOW'].iloc[i] > data['LOW'].iloc[i - 1]):
            ascending_triangles.append(i)

    # Identify the descending triangle
    descending_triangles.append(0)
    for i in range(1, len(data)):
        if (data['HIGH'].iloc[i] > highest_high.iloc[i] and
                data['LOW'].iloc[i] < lowest_low.iloc[i] and
                data['HIGH'].iloc[i] > data['HIGH'].iloc[i - 1] and
                data['LOW'].iloc[i] < data['LOW'].iloc[i - 1]):
            descending_triangles.append(i)

    # Check if the patterns are valid
    valid_patterns = []
    for i in range(1, len(ascending_triangles)):
        if (data['LOW'].iloc[ascending_triangles[i]] > data['LOW'].iloc[ascending_triangles[i - 1]]):
            valid_patterns.append(ascending_triangles[i])

    for i in range(1, len(descending_triangles)):
        if (data['HIGH'].iloc[descending_triangles[i]] < data['HIGH'].iloc[descending_triangles[i - 1]]):
            valid_patterns.append(descending_triangles[i])

    return valid_patterns


def is_pin_bar(candle):
    body_size = abs(candle['CLOSE'] - candle['OPEN'])
    wick_size = max(candle['HIGH'] - max(candle['CLOSE'], candle['OPEN']),
                    max(candle['LOW'], min(candle['OPEN'], candle['CLOSE'])) - candle['LOW'])
    return wick_size >= 2 * body_size


def identify_pin_bars(data):
    pin_bars = []
    for i in range(1, len(data)):
        if is_pin_bar(data.iloc[i]):
            pin_bars.append(i)
    return pin_bars

def is_bullish_engulfing(candle1, candle2):
    return (candle2['OPEN'] < candle1['CLOSE'] and
            candle2['CLOSE'] > candle1['OPEN'] and
            candle2['HIGH'] > candle1['HIGH'] and
            candle2['LOW'] < candle1['LOW'])


def is_bearish_engulfing(candle1, candle2):
    return (candle2['OPEN'] > candle1['CLOSE'] and
            candle2['CLOSE'] < candle1['OPEN'] and
            candle2['HIGH'] > candle1['HIGH'] and
            candle2['LOW'] < candle1['LOW'])


def identify_engulfing_candles(data):
    engulfing_candles = []
    for i in range(2, len(data)):
        if is_bullish_engulfing(data.iloc[i - 1], data.iloc[i]):
            engulfing_candles.append((i - 1, i, 'Bullish Engulfing'))
        elif is_bearish_engulfing(data.iloc[i - 1], data.iloc[i]):
            engulfing_candles.append((i - 1, i, 'Bearish Engulfing'))
    return engulfing_candles


def preprocess_data(file_path):
    data = pd.read_csv('C:/Users/User/Downloads/EURUSD_15.csv', sep='\t', header=None, skiprows=1)
    data.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']
    return data


def predict_buy_sell(data):
    # Extract relevant features
    features = data[['HIGH', 'LOW', 'CLOSE']][:-1]  # Exclude the last row

    # Create a binary target variable indicating whether the price will go up or down
    data['Price_Up'] = data['CLOSE'].shift(-1) > data['CLOSE']
    target = data['Price_Up'][:-1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')

    # Make predictions for the entire dataset
    all_predictions = model.predict(features)

    # Convert binary predictions to "Buy" or "Sell"
    buy_sell_predictions = ['Buy' if prediction else 'Sell' for prediction in all_predictions]

    # Print the predictions
    for prediction in buy_sell_predictions:
        print(prediction)

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



if __name__ == "__main__":
    # Load the CSV file
    data = pd.read_csv('C:/Users/User/Downloads/EURUSD_15.csv', sep='\t', header=None, skiprows=1)

    # Set column names
    data.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']

    # Call functions to identify patterns
    head_shoulders_pattern = head_and_shoulders(data)
    double_bottoms, double_tops = identify_patterns(data)
    wedge_patterns = identify_wedge_patterns(data)
    wedge_continuation_patterns = identify_wedge_continuation_patterns(data)
    flag_patterns = identify_flag_patterns(data)
    triangle_patterns = identify_triangle_patterns(data)
    pin_bars = identify_pin_bars(data)
    engulfing_candles = identify_engulfing_candles(data)

    # Generate signals for identified patterns
    patterns = {
        'Head and Shoulders': head_shoulders_pattern,
        'Double Bottoms': 'Sell' if double_bottoms else 'No Signal',
        'Double Tops': 'Buy' if double_tops else 'No Signal',
        'Wedge Patterns': 'Sell' if wedge_patterns else 'No Signal',
        'Wedge Continuation Patterns': 'Buy' if wedge_continuation_patterns else 'No Signal',
        'Flag Patterns': 'Sell' if flag_patterns else 'No Signal',
        'Triangle Patterns': 'Sell' if triangle_patterns else 'No Signal',
        'Pin Bars': 'Buy' if pin_bars else 'No Signal',
        'Engulfing Candles': 'Buy' if engulfing_candles else 'No Signal'
    }

    # Call the function to predict buy/sell using machine learning
    ml_predictions = predict_buy_sell(data)

    # Generate final signals by combining pattern signals and machine learning predictions
    final_signals = generate_signals(patterns)

    # Print the final signals
    for pattern, signal in final_signals.items():
        print(f'{pattern}: {signal}')


