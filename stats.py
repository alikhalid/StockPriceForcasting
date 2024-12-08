import numpy as np
import pandas as pd


def compute_prediction_stats(tickers, pred, obj):
    num_points = pred.shape[0]
    num_stocks = len(tickers)

    columns = ["Accuracy", "StdDevObj", "StdDevPred", "PredRealization", "RMSE"]
    stats = np.zeros((num_stocks, len(columns)))

    for stock_id in range(num_stocks):
        _pred = np.array(pred[:, stock_id])
        _obj = np.array(obj[:, stock_id])

        accuracy = np.sum((_pred * _obj) > 0)/np.sum(_pred != 0)
        std_dev_obj = np.sqrt(np.dot(_obj, _obj) / len(_obj))
        std_dev_pred = np.sqrt(np.dot(_pred, _pred) / len(_pred))
        realization = (np.dot(_pred, _obj) / len(_pred)) / std_dev_pred
        rmse = np.sqrt(np.mean(np.square(_pred - _obj)))

        stats[0, :] = np.array([accuracy, std_dev_obj, std_dev_pred, realization, rmse])

    df = pd.DataFrame(stats, columns=columns)
    df.insert(0, "Tickers", tickers)
    return df

def compute_profit_stats(tickers, pred, prices):
    pass
