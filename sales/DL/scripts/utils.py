import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()

    df_result = pd.DataFrame(data={"value": vals, "prediction": preds})
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


def calculate_metrics(df):
    return {
        "mae": mean_absolute_error(df.value, df.prediction),
        "rmse": mean_squared_error(df.value, df.prediction) ** 0.5,
        "r2": r2_score(df.value, df.prediction),
    }


def get_model_ckpt_name(run):
    return list(run.get_structure()["training"]["model"]["checkpoints"].keys())[-1]
