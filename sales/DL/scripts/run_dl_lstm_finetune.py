from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from neptune.new.types import File
import pytorch_lightning as pl
import seaborn as sns
from data_module import *
from model import *
from utils import *


def main():
    params = {
        "seq_len": 8,
        "batch_size": 128,
        "criterion": nn.MSELoss(),
        "max_epochs": 1,
        "n_features": 1,
        "hidden_dim": 512,
        "n_layers": 5,
        "dropout": 0.2,
        "learning_rate": 0.001,
    }

    # (neptune) Download model checkpoint from model registry
    model_version = neptune.init_model_version(
        model="TSF-DL",
        with_id="TSF-DL-1",
    )

    model_version["checkpoint"].download()

    # (neptune) Create NeptuneLogger instance
    neptune_logger = NeptuneLogger()

    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        callbacks=[early_stop, lr_logger],
        logger=neptune_logger # neptune integration
    )

    dm = WalmartSalesDataModule(
        seq_len=params["seq_len"], num_workers=8, path="./sales/data/aggregate_data.csv"
    )

    model = LSTMRegressor(
        n_features=params["n_features"],
        hidden_dim=params["hidden_dim"],
        criterion=params["criterion"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
    )

    model = model.load_from_checkpoint('checkpoint.ckpt')

    # Train model
    trainer.fit(model, dm)

    # Test model
    test_loader = dm.test_dataloader()
    predictions, values = model.predict(test_loader)
    df_result = format_predictions(predictions, values, dm.df, dm.scaler)

    preds_plot = sns.lineplot(data=df_result)

     # (neptune) Log predictions visualizations
    neptune_logger.experiment["training/plots/ypred_vs_y_valid"].upload(File.as_image(preds_plot.figure))

    val_metrics = calculate_metrics(df_result)

    # (neptune) Log scores
    neptune_logger.experiment["training/val"] = val_metrics


if __name__ == "__main__":
    main()
