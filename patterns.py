import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf

# Fetch 15-minute interval data from Yahoo Finance
def fetch_data(symbol='AUDUSD=X', interval='15m'):
    data = yf.download(tickers=symbol, interval=interval, period="60d")
    data.reset_index(inplace=True)
    data.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'ADJ CLOSE', 'VOLUME']
    data.drop(columns=['ADJ CLOSE', 'VOLUME'], inplace=True)
    return data


def head_and_shoulders(data, lookback=50):
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

def identify_wedge_patterns(data, window=50):
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

def identify_wedge_continuation_patterns(data, window=50):
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

def identify_flag_patterns(data, window=50):
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

def identify_triangle_patterns(data, window=50):
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

# Moving Average function
def moving_average(data, window):
  return data['CLOSE'].rolling(window=window).mean()

# Detect Round Bottom pattern
def detect_round_bottom(data, window=50):
    ma = moving_average(data, window)
    data['RoundBottom'] = ((ma.shift(window) > ma) & (ma.shift(window // 2) < ma) & (ma > ma.shift(-window))).astype(int)
    return data

def detect_round_top(data, window=50):
    ma = moving_average(data, window)
    data['RoundTop'] = ((ma.shift(window) < ma) & (ma.shift(window // 2) > ma) & (ma < ma.shift(-window))).astype(int)
    return data

# Detect Cup and Handle pattern
def detect_cup_handle(data, window=50, handle_length=10):
    ma = moving_average(data, window)
    data['CupHandle'] = ((ma.shift(window) > ma) & (ma.shift(window // 2) < ma) & (ma > ma.shift(-window)) &
                         (data['CLOSE'].shift(-handle_length) > data['CLOSE'])).astype(int)
    return data


# Detect Inverse Cup and Handle pattern
def detect_inverse_cup_handle(data, window=50, handle_length=10):
    ma = moving_average(data, window)
    data['InverseCupHandle'] = ((ma.shift(window) < ma) & (ma.shift(window // 2) > ma) & (ma < ma.shift(-window)) &
                                (data['CLOSE'].shift(-handle_length) < data['CLOSE'])).astype(int)
    return data

def predict_buy_sell(data):
    data['RETURN'] = data['CLOSE'].pct_change()
    data.dropna(inplace=True)

    X = data[['OPEN', 'HIGH', 'LOW', 'CLOSE']]
    y = (data['RETURN'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    data['PREDICTION'] = model.predict(X)

    return data, accuracy

def evaluate_patterns(patterns, data):
    accuracies = {}
    for pattern_name, indices in patterns.items():
        if len(indices) == 0:
            accuracies[pattern_name] = 0
            continue

        signals = np.zeros(len(data))
        signals[indices] = 1

        returns = data['CLOSE'].pct_change().shift(-1)
        signals = signals[:len(returns)]
        returns = returns.dropna()
        signals = signals[:len(returns)]

        accuracy = ((signals == (returns > 0)).sum()) / len(signals)
        accuracies[pattern_name] = accuracy

    return accuracies

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

    # Detect additional patterns
    data = detect_round_bottom(data)
    data = detect_round_top(data)
    data = detect_cup_handle(data)
    data = detect_inverse_cup_handle(data)

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
        'Bearish Engulfing': bearish_engulfing,
        'Round Bottom': data[data['RoundBottom'] == 1].index.tolist(),
        'Round Top': data[data['RoundTop'] == 1].index.tolist(),
        'Cup and Handle': data[data['CupHandle'] == 1].index.tolist(),
        'Inverse Cup and Handle': data[data['InverseCupHandle'] == 1].index.tolist()
    }

    # Evaluate patterns
    accuracies = evaluate_patterns(patterns, data)
    print("Pattern Accuracies:", accuracies)

    # Predict buy/sell using machine learning
    ml_predictions, ml_accuracy = predict_buy_sell(data)

    # Find the pattern with the highest accuracy
    best_pattern = max(accuracies, key=accuracies.get)
    print(f'Best Pattern: {best_pattern} with Accuracy: {accuracies[best_pattern] * 100:.2f}%')

    # Generate final signal based on the best pattern
    best_pattern_indices = patterns[best_pattern]
    final_signal = 'Buy' if len(best_pattern_indices) > 0 and data['CLOSE'].iloc[best_pattern_indices[-1]] > data['OPEN'].iloc[best_pattern_indices[-1]] else 'Sell'

    print(f'Final Trading Signal: {final_signal}')

if __name__ == "__main__":
    main()