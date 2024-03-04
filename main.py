import MetaTrader5 as mt5
from patterns import *
import pandas as pd

# mt5 credentials
login = 130798
password = 'Mare-Dewy-09'
server = 'EGMSecurities-Demo'


# Connect to the MetaTrader 5 terminal
def connect_to_mt5(login, password, server):
    # Connect to the MetaTrader 5 terminal
    if not mt5.initialize(login=login, password=password, server=server):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

# Your trading bot logic goes here
symbol = "EURUSD"
lot = 0.01
deviation = 20

# Load historical data
data = pd.read_csv('C:/Users/User/Downloads/EURUSD_15.csv', sep='\t', header=None, skiprows=1)
data.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']


# Generate trading signals from patterns
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
        'Double Bottoms': double_bottoms,
        'Double Tops': double_tops,
        'Wedge Patterns': wedge_patterns,
        'Wedge Continuation Patterns': wedge_continuation_patterns,
        'Bull Flags': bull_flags,
        'Bear Flags': bear_flags,
        'Ascending Triangles': asc_triangles,
        'Descending Triangles': desc_triangles,
        'Pin Bars': pin_bars,
        'Engulfing Candles': engulfing_candles
    }
# Predict buy/sell signals using machine learning
ml_predictions = predict_buy_sell(data)

# Generate signals based on patterns and ML predictions
final_signals = generate_signals(patterns)


def execute_trade(symbol, lot, deviation, final_signals):

    price = mt5.symbol_info_tick(symbol).ask

    # Get the current tick information
    point = mt5.symbol_info(symbol).point


    order_type = mt5.ORDER_TYPE_BUY if final_signals == 'buy' else mt5.ORDER_TYPE_SELL


    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": "Python script open order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

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

# Connect to MetaTrader 5
connect_to_mt5(login, password, server)

# Execute trading logic
execute_trade(symbol, lot, deviation, final_signals)

# Shutdown MetaTrader 5 connection
shutdown_mt5()
