import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

np.random.seed(7)
# importing datasets
data = pd.read_csv('AAPL Historical Data.csv')
usecols = [1, 2, 3, 4]
data = data.reindex(index=data.index[::-1])

obs = np.arange(1, len(data) + 1, 1)
OHLC_avg = data.mean(axis=1)
data.head()
