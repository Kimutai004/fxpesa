import MetaTrader5 as mt5
import yfinance as yf
from patterns import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# MetaTrader 5 credentials
login = 130798
password = 'Mare-Dewy-09'
server = 'EGMSecurities-Demo'


# Connect to the MetaTrader 5 terminal
def connect_to_mt5(login, password, server):
    if not mt5.initialize(login=login, password=password, server=server):
        print("initialize() failed, error code =", mt5.last_error())
        quit()


# Define trading parameters
symbol = "AUDUSD"
lot = 0.1
deviation = 20


def get_open_trades(symbol):
    """Get all open trades for the given symbol."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        print(f"No positions for {symbol}, error code={mt5.last_error()}")
        return []
    return positions


def close_trade(position):
    """Close the given position."""
    if position.profit <= 0:
        return False

    if position.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(position.symbol).bid
    elif position.type == mt5.ORDER_TYPE_SELL:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": "Python script close order",
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close position {position.ticket}, retcode={result.retcode}")
        return False
    else:
        print(f"Position {position.ticket} closed successfully")
        return True


def execute_trade(symbol, lot, deviation, final_signal):
    # Get the current tick information
    price = mt5.symbol_info_tick(symbol).ask
    point = mt5.symbol_info(symbol).point

    # Determine the order type and set the order price
    if 'Buy' in final_signal:
        if 'Limit' in final_signal:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
            order_price = price  # Adjust this as per your strategy for buy limit price
        else:
            order_type = mt5.ORDER_TYPE_BUY
            order_price = price
    elif 'Sell' in final_signal:
        if 'Limit' in final_signal:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT
            order_price = price  # Adjust this as per your strategy for sell limit price
        else:
            order_type = mt5.ORDER_TYPE_SELL
            order_price = price

    # Set take profit and stop loss
    take_profit = order_price + 500 * point if 'Buy' in final_signal else order_price - 500 * point
    stop_loss = order_price - 500 * point if 'Buy' in final_signal else order_price + 500 * point

    # Create order request
    request = {
        "action": mt5.TRADE_ACTION_PENDING if 'Limit' in final_signal else mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": order_price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": deviation,
        "magic": 234000,
        "comment": "Python script open order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    # Send the order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode={result.retcode}")
        print(result.comment)
    else:
        print("Order placed successfully")
        print(f"Order: ticket={result.order}")


# Shutdown connection to the MetaTrader 5 terminal
def shutdown_mt5():
    mt5.shutdown()


class TradeManager:
    def __init__(self):
        self.last_signal_type = None

    def manage_trades(self, symbol, lot, deviation, final_signal):
        current_signal_type = 'Buy' if 'Buy' in final_signal else 'Sell'

        if current_signal_type != self.last_signal_type:
            open_trades = get_open_trades(symbol)
            for trade in open_trades:
                if not close_trade(trade):
                    print(f"Skipping close for position {trade.ticket}")

            execute_trade(symbol, lot, deviation, final_signal)
            self.last_signal_type = current_signal_type


def fetch_data(symbol='AUDUSD=X', interval='15m'):
    data = yf.download(tickers=symbol, interval=interval, period="60d")
    data.reset_index(inplace=True)
    data.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'ADJ CLOSE', 'VOLUME']
    data.drop(columns=['ADJ CLOSE', 'VOLUME'], inplace=True)
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

    return data


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


def combine_signals(patterns, ml_predictions, data):
    # Evaluate patterns
    accuracies = evaluate_patterns(patterns, data)
    print("Pattern Accuracies:", accuracies)

    # Find the pattern with the highest accuracy
    best_pattern = max(accuracies, key=accuracies.get)
    print(f'Best Pattern: {best_pattern} with Accuracy: {accuracies[best_pattern] * 100:.2f}%')

    # Generate final signal based on the best pattern
    best_pattern_indices = patterns[best_pattern]
    final_signal = 'Buy' if len(best_pattern_indices) > 0 and data['CLOSE'].iloc[best_pattern_indices[-1]] > \
                            data['OPEN'].iloc[best_pattern_indices[-1]] else 'Sell'

    return final_signal


def main():
    # Connect to MetaTrader 5
    connect_to_mt5(login, password, server)

    # Fetch data and generate signals
    data = fetch_data()
    ml_predictions = predict_buy_sell(data)

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

    # Combine signals
    final_signal = combine_signals(patterns, ml_predictions, data)

    # Initialize TradeManager and manage trades
    trade_manager = TradeManager()
    trade_manager.manage_trades(symbol, lot, deviation, final_signal)

    # Shutdown MetaTrader 5 connection
    shutdown_mt5()


if __name__ == "__main__":
    main()
