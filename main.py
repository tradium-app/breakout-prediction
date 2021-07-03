# %% Import libraries
import numpy as np
import pandas as pd
import random as rn
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import plotly.express as px

rn.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
plt.style.use("fivethirtyeight")
matplotlib.rcParams["figure.figsize"] = (20, 10)

# %% Import data
dtypes = [
    ("Date", datetime),
    ("Open", float),
    ("High", float),
    ("Low", float),
    ("Close", float),
    ("Volume", float),
    ("OpenInt", float),
]
df = pd.read_csv("./data/qqq.us.txt", dtype=dtypes, index_col=0, parse_dates=True)

df = df.drop(["OpenInt"], axis=1)

# df["Close"].plot()
# %% Prepare Prediction column
def willBreakOut(x):
    if (x.iloc[1:] > x[0] * 1.05).any():
        return 1
    elif (x.iloc[1:] < x[0] * 0.95).any():
        return -1
    else:
        return 0


df["BreakOut"] = df["Close"].rolling(window=10).apply(willBreakOut)
df.fillna(0, inplace=True)
df.plot()

#%% Get Compute the number of rows to train the model on
training_data_len = int(len(df) * 0.8)

# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

batch_size = 60
pd.DataFrame(scaled_data).plot()

# %% Define helpful functions
def create_batch_dataset(dataframe, batch_size=60):
    x_train, y_train = [], []
    ROLLING_WINDOW = 12

    for i in range(batch_size, len(dataframe) - ROLLING_WINDOW):
        x_train.append(dataframe[i - batch_size : i, :])
        y_train.append(dataframe[i, -1:])

    return np.array(x_train), np.array(y_train)


def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


#%% Build the LSTM network model
x_train, y_train = create_batch_dataset(train_data, batch_size)

model = Sequential()
model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(x_train.shape[1], x_train.shape[2]),
    )
)
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

#%% Train the model
stoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

history = model.fit(
    x_train, y_train, batch_size, epochs=3, callbacks=[stoppingCallback], shuffle=False,
)

# %% prepare test data
testing_data_len = len(df) - training_data_len
test_data = scaled_data[training_data_len:, :]

x_test, y_test = create_batch_dataset(test_data, batch_size)

# %% Run prediction

x_train_predict = model.predict(x_train, batch_size=batch_size)
model.reset_states()
x_test_predict = model.predict(x_test, batch_size=batch_size)

cols = ["Open", "High", "Low", "Close", "Volume", "BreakOut"]

x_train_predict = invTransform(scaler, x_train_predict, "BreakOut", cols)
y_train = invTransform(scaler, y_train, "BreakOut", cols)
x_test_predict = invTransform(scaler, x_test_predict, "BreakOut", cols)
y_test = invTransform(scaler, y_test, "BreakOut", cols)


# calculate root mean squared error
trainScore = np.math.sqrt(mean_squared_error(y_train, x_train_predict))
print("Train Score: %.2f RMSE" % (trainScore))
testScore = np.math.sqrt(mean_squared_error(y_test, x_test_predict))
print("Test Score: %.2f RMSE" % (testScore))


# %% Plot results
# shift train predictions for plotting
result_df = df.copy()
# result_df.loc[:, "Close"] = scaled_data[:, 0]
result_df.iloc[:, :-1] = scaled_data[:, :-1]

# x_train_predict[[x_train_predict > 0]]

result_df["training_prediction"] = np.nan
result_df.iloc[batch_size : len(x_train_predict) + batch_size, -1:] = np.reshape(
    x_train_predict, (len(x_train_predict), 1)
)
result_df.loc[
    (result_df["training_prediction"] < 0.25)
    & (result_df["training_prediction"] > -0.25),
    "training_prediction",
] = 0

result_df["test_prediction"] = np.nan
result_df.iloc[-len(x_test_predict) :, -1:] = np.reshape(
    x_test_predict, (len(x_test_predict), 1)
)
result_df.loc[
    (result_df["test_prediction"] < 0.25) & (result_df["test_prediction"] > -0.25),
    "test_prediction",
] = 0


# sns.catplot(data=result_df.loc["2000-01-01":"2000-03-01"])
px.scatter(
    result_df.loc["1990-01-01":"2022-03-01"][
        ["BreakOut", "training_prediction", "test_prediction"]
    ],
    width=1200,
    height=800,
)

