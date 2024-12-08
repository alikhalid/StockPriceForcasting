import torch

import yfinance as yf
import numpy as np


def load_stock(ticker):
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

    obj = (next_close - next_open) / next_open
    data = np.concatenate([[(close - open) / open],
                           [(next_open - close) / close],
                           [vol]])
    return data.astype(np.float32)[:, :1000], obj.astype(np.float32)[:1000]


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
