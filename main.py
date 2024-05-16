import MetaTrader5 as mt5
from patterns import *

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
symbol = "EURUSD"
lot = 0.1
deviation = 20

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
    take_profit = order_price + 500 * point if 'Buy' in final_signal else order_price - 50 * point
    stop_loss = order_price - 500 * point if 'Buy' in final_signal else order_price + 50 * point

    # Create order request
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
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
    final_signal = combine_signals(patterns, ml_predictions, data)  # Pass 'data' as the third argument
    
    # Execute trade based on final signal
    execute_trade(symbol, lot, deviation, final_signal)

    # Shutdown MetaTrader 5 connection
    shutdown_mt5()

if __name__ == "__main__":
    main()
