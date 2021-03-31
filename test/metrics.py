import pandas as pd
from scipy.stats import spearmanr
from sklearn import metrics
import numpy as np


def get_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features by calculating statistical attributes for each group.
    :param df: Pandas DataFrame containing all features
    :return: Pandas dataframe with stats columns appended
    """
    for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
        cols = [col for col in df.columns if group in col]
        df[f"feature_{group}_mean"] = df[cols].mean(axis=1)
        df[f"feature_{group}_std"] = df[cols].std(axis=1)
        df[f"feature_{group}_skew"] = df[cols].skew(axis=1)
    return df


def sharpe_ratio(corrs: pd.Series) -> np.float32:
    """
    Calculate the Sharpe ratio for Numerai by using grouped per-era data

    :param corrs: A Pandas Series containing the Spearman correlations for each era
    :return: A float denoting the Sharpe ratio of your predictions.
    """
    return corrs.mean() / corrs.std()


def evaluate(df: pd.DataFrame) -> tuple:
    """
    Evaluate and display relevant metrics for Numerai
    :rtype: object
    :param df: A Pandas DataFrame containing the columns "era", "target" and "prediction"
    :return: A tuple of float containing the metrics
    """
    def _score(sub_df: pd.DataFrame) -> np.float32:
        """Calculates Spearman correlation"""
        return spearmanr(sub_df["target"], sub_df["prediction"])[0]

    # Calculate metrics
    corrs = df.groupby("era").apply(_score)
    payout_raw = (corrs / 0.2).clip(-1, 1)
    spearman = round(corrs.mean(), 4)
    payout = round(payout_raw.mean(), 4)
    numerai_sharpe = round(sharpe_ratio(corrs), 4)
    mae = metrics.mean_absolute_error(df["target"], df["prediction"]).round(4)
    mse = metrics.mean_squared_error(df["target"], df["prediction"])
    rmse = np.sqrt(metrics.mean_squared_error(df["target"], df["prediction"]))
    mape = np.mean(np.abs((df["target"] - df["prediction"]) / np.abs(df["target"])))

    # Display metrics
    print(f"Spearman Correlation: {spearman}")
    print(f"Average Payout: {payout}")
    print(f"Sharpe Ratio: {numerai_sharpe}")
    print(f"Mean Absolute Error (mae): {mae}")
    print(f"Mean Squared Error (mse): {mse}")
    print(f"Root Mean Squared Error (rmse): {rmse}")
    print('Mean Absolute Percentage Error (mape): ', round(mape * 100, 2))
    print('Accuracy:', round(100*(1 - mape), 2))

    return spearman, payout, numerai_sharpe, mae, mse, rmse, mape

# ######################################################################################################################

# TODO: Numerai Compute automation
