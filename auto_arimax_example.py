import neptune.new as neptune
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.feature_exgineering import prepare_features, exogenous_features

np.random.seed(9473)
path_to_data = "data/BAJAJFINSV.csv"
stock_name = "BAJAJFINSV"

# (neptune) create run
run = neptune.init(project="common/project-time-series-forecasting",
                   tags=["arimax"])

# (neptune) log stock name to the run
run["info/stock_name"] = stock_name

# load time series data (stock prices)
df = pd.read_csv(path_to_data)
df.set_index("Date", drop=False, inplace=True)

ax = df.VWAP.plot(figsize=(9, 6))
ax.grid("both")

# (neptune) log VWAP chart as an interactive visualization
run["visualizations/VWAP_chart"] = neptune.types.File.as_image(ax.figure)
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

# fit arima
model = auto_arima(
    df_train.VWAP,
    exogenous=df_train[exogenous_features],
    trace=True,
    error_action="ignore",
    suppress_warnings=True
)
model.fit(df_train.VWAP, exogenous=df_train[exogenous_features])
forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast_ARIMAX"] = forecast

ax = df_valid[["VWAP", "Forecast_ARIMAX"]].plot(figsize=(9, 6))
ax.grid("both")

# (neptune) log data and forecast as an interactive chart
run["visualizations/VWAP-forecast"] = neptune.types.File.as_html(ax.figure)
plt.close("all")

# (neptune) log final metrics
run["valid/rmse"] = np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))
run["valid/mae"] = mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)
