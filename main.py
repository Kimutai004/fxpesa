# Import necessary libraries
import MetaTrader5 as mt5

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
# Function to calculate moving averages
def calculate_moving_average(symbol, timeframe, ma_period):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, ma_period)
    if rates is not None and len(rates) > 0:
        sum_close = sum(rate['close'] for rate in rates)
        return sum_close / len(rates)
    else:
        return None

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(symbol, timeframe, rsi_period):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, rsi_period)
    if rates is not None and len(rates) > 0:
        price_changes = [rates[i + 1]['close'] - rates[i]['close'] for i in range(len(rates) - 1)]
        positive_changes = [change for change in price_changes if change > 0]
        negative_changes = [-change for change in price_changes if change < 0]

        avg_gain = sum(positive_changes) / rsi_period
        avg_loss = sum(negative_changes) / rsi_period

        if avg_loss == 0:
            return 100
        else:
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
    else:
        return None


# Implement your trading strategy here
def trading_strategy(symbol, timeframe, ma_period, rsi_period, macd_fast_period, macd_slow_period, macd_signal_period):
    # Calculate indicators
    ma = calculate_moving_average(symbol, timeframe, ma_period)
    rsi = calculate_rsi(symbol, timeframe, rsi_period)

    # Get the current price of the symbol
    tick = mt5.symbol_info_tick(symbol)
    if tick is not None:
        price = tick.bid

        if ma is not None and rsi is not None :
            if price > ma and rsi < 30:
                # Execute Buy trade
                execute_trade(symbol, lot, deviation, price)
                pass
            elif price < ma and rsi > 70:
                execute_trade(symbol, lot, deviation)
                pass
            else:
                # Hold position or do nothing
                pass
    else:
        print("Failed to get the current price of the symbol.")


def execute_trade(symbol, lot, deviation):

    price = mt5.symbol_info_tick(symbol).ask

    # Get the current tick information
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask

    stop_loss_pips = 50  # 50 pips take profit
# Calculate stop-loss and take-profit levels
    stop_loss = price - (stop_loss_pips * point)
    take_profit = price + (stop_loss_pips * point)



    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": "Python script open buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "sl": stop_loss,
        "tp": take_profit

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
execute_trade(symbol, lot, deviation)
trading_strategy(symbol, mt5.TIMEFRAME_M1, 20, 14, 12, 26, 9)

# Shutdown MetaTrader 5 connection
shutdown_mt5()
