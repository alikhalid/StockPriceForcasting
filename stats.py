import numpy as np

class Profit:
    def __init__(self, time, prices, num_points):
        self.time = time
        self.prices = prices

        self.num_points = int(min(len(prices) * 0.5, num_points))
        self.profit = np.zeros(10)

    def compute(self, time, buy):

