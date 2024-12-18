import torch

import yfinance as yf
import numpy as np


def load_stock_yfinance(ticker, num_t=500):
    ticker = yf.Ticker(ticker)
    historical_data = ticker.history(period="5y")
    close = np.array(historical_data.Close)
    open = np.array(historical_data.Open)

    next_open = open[2:]
    next_close = close[2:]
    last_open = open[:-2]
    last_close = close[:-2]
    open = open[1:-1]
    close = close[1:-1]

    vol = np.array(historical_data.Volume)[1:-1]
    vol = vol / np.mean(vol)

    obj1 = (next_close - close) / close
    obj2 = (next_open - close) / close
    data = np.concatenate([[(close - open) / open],
                           [(next_open - close) / close],
                           #[(open - last_close) / last_close],
                           [vol]])
    return data.astype(np.float32)[:, :num_t], obj1.astype(np.float32)[:num_t]



def load_stock_intraday(ticker):
    ticker = yf.Ticker(ticker)
    historical_data = ticker.history(period="5d", interval="5m")
    close = np.array(historical_data.Close)
    open = np.array(historical_data.Open)

    next_open = open[2:]
    next_close = close[2:]
    last_open = open[:-2]
    last_close = close[:-2]
    open = open[1:-1]
    close = close[1:-1]

    vol = np.array(historical_data.Volume)[1:-1]
    vol = vol / np.mean(vol)

    obj1 = (next_close - next_open) / next_open
    obj2 = (next_open - close) / close
    data = np.concatenate([[(close - open) / open],
                           [(next_open - close) / close],
                           [vol]])
    return data.astype(np.float32)[:, :1000], obj1.astype(np.float32)[:1000]


def setup_cov_tensors(X, y, sequence_length):
    targets = y[..., sequence_length:]
    inputs = []
    for i in range(y.shape[-1] - sequence_length):
        inputs.append(X[..., i:i + sequence_length])

    inputs = np.array(inputs)

    # Convert to PyTorch tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    return inputs, targets.transpose(0, 1)
