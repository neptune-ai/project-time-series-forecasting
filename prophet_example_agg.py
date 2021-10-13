import neptune.new as neptune
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import prophet_utils
from src.feature_exgineering import prepare_features, exogenous_features

np.random.seed(9476)
path_to_data = "data/BAJAJFINSV.csv"
stock_name = "BAJAJFINSV"

CONFIG = {
    "changepoint_prior_scale": 0.5,
    "seasonality_prior_scale": 10.0,
    "daily_seasonality": True,
}

# (neptune) create run
run = neptune.init(
    project="common/project-time-series-forecasting",
    tags=["prophet", "agg"],
    name="prophet-grid-search-run",
)

# (neptune) track data version
run["data"].track_files(path_to_data)

# (neptune) log stock name to the run
run["info/stock_name"] = stock_name

# load time series data (stock prices)
df = pd.read_csv(path_to_data)
df.set_index("Date", drop=False, inplace=True)

ax = df.VWAP.plot(figsize=(9, 6))
ax.grid("both")

# (neptune) log VMAP chart as an interactive visualization
run["visualizations/VMAP_chart"] = neptune.types.File.as_image(ax.figure)
plt.close("all")

# feature engineering
df_processed = prepare_features(df)

# train/valid split
valid_date_split = "2019"
df_train = df_processed[df_processed.Date < valid_date_split]
df_valid = df_processed[df_processed.Date >= valid_date_split]

# (neptune) log year for train/valid split
run["valid/split"] = valid_date_split

# (neptune) log exogenous feature names
run["feature_engineering/exogenous_features"] = exogenous_features

# fit Facebook Prophet model
model = Prophet(**CONFIG)
for feature in exogenous_features:
    model.add_regressor(feature)

# (neptune) log model config
prophet_utils.log_config(run, model)

model.fit(df_train[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds", "VWAP": "y"}))

# (neptune) log model params
prophet_utils.log_params(run, model)

forecast = model.predict(df_valid[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds"}))
df_valid["Forecast_Prophet"] = forecast.yhat.values

# (neptune) log forecast plots (can be interactive)
prophet_utils.log_forecast_plots(run, model, forecast, log_interactive=True)

ax = df_valid[["VWAP", "Forecast_Prophet"]].plot(figsize=(9, 6))
ax.grid("both")

# (neptune) log data and forecast as an interactive chart
run["visualizations/VMAP-forecast"] = neptune.types.File.as_html(ax.figure)
plt.close("all")

# (neptune) log final metrics
run["valid/prophet/rmse"] = np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_Prophet))
run["valid/prophet/mae"] = mean_absolute_error(df_valid.VWAP, df_valid.Forecast_Prophet)
