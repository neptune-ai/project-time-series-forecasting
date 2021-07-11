import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


exogenous_features = ["High_mean_lag3", "High_std_lag3", "Low_mean_lag3", "Low_std_lag3",
                      "Volume_mean_lag3", "Volume_std_lag3", "Turnover_mean_lag3",
                      "Turnover_std_lag3", "Trades_mean_lag3", "Trades_std_lag3",
                      "High_mean_lag7", "High_std_lag7", "Low_mean_lag7", "Low_std_lag7",
                      "Volume_mean_lag7", "Volume_std_lag7", "Turnover_mean_lag7",
                      "Turnover_std_lag7", "Trades_mean_lag7", "Trades_std_lag7",
                      "High_mean_lag30", "High_std_lag30", "Low_mean_lag30", "Low_std_lag30",
                      "Volume_mean_lag30", "Volume_std_lag30", "Turnover_mean_lag30",
                      "Turnover_std_lag30", "Trades_mean_lag30", "Trades_std_lag30",
                      "month", "week", "day", "day_of_week"]


def prepare_features(df):
    df.reset_index(drop=True, inplace=True)
    lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
    window1 = 3
    window2 = 7
    window3 = 30

    df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
    df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
    df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

    df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
    df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
    df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

    df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
    df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
    df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

    for feature in lag_features:
        df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
        df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
        df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]

        df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
        df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
        df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

    df.fillna(df.mean(), inplace=True)
    df.set_index("Date", drop=False, inplace=True)

    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df["month"] = df.Date.dt.month
    df["week"] = df.Date.dt.week
    df["day"] = df.Date.dt.day
    df["day_of_week"] = df.Date.dt.dayofweek

    return df


def prepare_data_for_lstm(df, valid_date_split):
    features = ["Date", "VWAP"]
    df_processed = df[features]

    values = df_processed.VWAP.values
    values = values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    df_processed["VWAP"] = scaled_data

    df_train = df_processed[df_processed.Date < valid_date_split]
    df_test = df_processed[df_processed.Date >= valid_date_split]

    df_train.index = df_train.Date
    df_train.drop('Date', axis=1, inplace=True)

    df_test.index = df_test.Date
    df_test.drop('Date', axis=1, inplace=True)
    df_test["VWAP"] = scaler.inverse_transform(df_test.values)

    np_train = df_train.values
    np_test = df_test.values

    x_train, y_train = [], []
    for i in range(90, len(np_train)):
        x_train.append(scaled_data[i - 90:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    inputs = scaled_data[len(scaled_data) - len(np_test) - 90:]
    x_test = []
    for i in range(90, len(inputs)):
        x_test.append(inputs[i - 90:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, scaler, df_test

