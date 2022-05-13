import glob

import neptune.new as neptune
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.feature_exgineering import prepare_data_for_lstm
from src.model import get_model

np.random.seed(9476)
path_to_data = "data/BAJAJFINSV.csv"
stock_name = "BAJAJFINSV"

params = {
    "loss": "mean_squared_error",
    "optimizer": "adam",
    "dropout": 0.2,
    "lstm_units": 30,
    "epochs": 10,
    "batch_size": 128
}

# (neptune) create run
run = neptune.init(project="common/project-time-series-forecasting",
                   tags=["lstm", "keras"])

# (neptune) log model params
run["LSTM/params"] = params

# (neptune) log stock name to the run
run["info/stock_name"] = stock_name

# load time series data (stock prices)
df = pd.read_csv(path_to_data)
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

ax = df.VWAP.plot(figsize=(9, 6))
ax.grid("both")

# (neptune) log VWAP chart as an static visualization
run["visualizations/VWAP_chart"] = neptune.types.File.as_image(ax.figure)
plt.close("all")

# feature engineering and train/valid split
valid_date_split = "2019"
x_train, y_train, x_test, scaler, df_test = prepare_data_for_lstm(df, valid_date_split)

# (neptune) log year for train/valid split
run["valid/split"] = valid_date_split

model = get_model(params=params, input_shape=x_train.shape[1])

# (neptune) use TF-Keras integration to log training metrics
neptune_callback = NeptuneCallback(run, base_namespace="training")

model.fit(
    x_train,
    y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    verbose=1,
    callbacks=[neptune_callback]
)

preds = model.predict(x_test)
preds = scaler.inverse_transform(preds)
df_test["Forecast_LSTM"] = preds

ax = df_test[["VWAP", "Forecast_LSTM"]].plot(figsize=(9, 6))
ax.grid("both")

# (neptune) log data and forecast as an interactive chart
run["visualizations/VWAP-forecast"] = neptune.types.File.as_html(ax.figure)
plt.close("all")

# (neptune) log final metrics
run["valid/rmse"] = np.sqrt(mean_squared_error(df_test.VWAP, df_test.Forecast_LSTM))
run["valid/mae"] = mean_absolute_error(df_test.VWAP, df_test.Forecast_LSTM)

# (neptune) save model weights
model.save('model_weights')
run["LSTM/model_weights/saved_model.pb"].upload('model_weights/saved_model.pb')
for name in glob.glob('model_weights/variables/*'):
    run["LSTM/{}".format(name)].upload(name)
