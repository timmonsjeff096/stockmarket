from sklearn.ensemble import RandomForestRegressor
import data_processing
import logging
import numpy as np
import pandas as pd

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"


# Submissions are scored by spearman correlation
def correlation(prediction, targets):
    ranked_preds = prediction.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])


def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)


def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                          feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda y: correlation(y["neutral_sub"], y[TARGET_NAME])).mean()
    return np.mean(scores)


def unif(df):
    z = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(z, index=df.index)


def neutralize(df,
               columns,
               extra_neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        logging.info(u)
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for n in scores.T:
                n = (pd.Series(n).rank(method="first").values - .5) / len(n)
                scores2.append(n)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


def neutralize_series(n_series, by, proportion=1.0):
    scores = n_series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(n_series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=n_series.index)
    return neutralized


class RandomForestRegression:
    def __init__(self):
        self.data = data_processing.DataProcessing()
        data_tup = self.data.get_data()
        self.train_data = data_tup[0]
        self.tournament_data = data_tup[1]

    def feature_names(self) -> list:
        features = [f for f in self.train_data.columns if f.startswith("feature")]
        logging.info(f"Loaded {len(features)} features")
        self.train_data[features] = self.train_data[features].astype(np.float32)
        self.tournament_data[features] = self.tournament_data[features].astype(np.float32)
        return features


rfr = RandomForestRegression()
model = RandomForestRegressor(n_estimators=20, max_depth=2, n_jobs=None, random_state=0, verbose=True)
feature_names = rfr.feature_names()
model.fit(rfr.train_data[feature_names], rfr.train_data[TARGET_NAME])

# Generate predictions on both training and tournament data
logging.info("Generating predictions...")
rfr.train_data[PREDICTION_NAME] = model.predict(rfr.train_data[feature_names])
rfr.tournament_data[PREDICTION_NAME] = model.predict(rfr.tournament_data[feature_names])

# Check the per-era correlations on the training set (in sample)
train_correlations = rfr.train_data.groupby("era").apply(score)
logging.info(f"On training the correlation has mean {train_correlations.mean()} and"
             f" std {train_correlations.std(ddof=0)}")
logging.info(f"On training the average per-era payout is {payout(train_correlations).mean()}")

validation_data = rfr.tournament_data[rfr.tournament_data.data_type == "validation"]
validation_correlations = validation_data.groupby("era").apply(score)
logging.info(f"On validation the correlation has mean {validation_correlations.mean()} and "
             f"std {validation_correlations.std(ddof=0)}")
logging.info(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

# Check the "sharpe" ratio on the validation set
validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
logging.info(f"Validation Sharpe: {validation_sharpe}")

logging.info("checking max drawdown...")
rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                              min_periods=1).max()
daily_value = (validation_correlations + 1).cumprod()
max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
logging.info(f"max drawdown: {max_drawdown}")

# Check the feature exposure of your validation predictions
feature_exposures = validation_data[feature_names].apply(lambda d: correlation(validation_data[PREDICTION_NAME], d),
                                                         axis=0)
max_per_era = validation_data.groupby("era").apply(
    lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
max_feature_exposure = max_per_era.mean()
logging.info(f"Max Feature Exposure: {max_feature_exposure}")

# Check feature neutral mean
logging.info("Calculating feature neutral mean...")
feature_neutral_mean = get_feature_neutral_mean(validation_data)
logging.info(f"Feature Neutral Mean is {feature_neutral_mean}")

rfr.tournament_data[PREDICTION_NAME].to_csv("submission.csv", header=True)
