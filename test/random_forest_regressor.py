from sklearn.ensemble import RandomForestRegressor
import data_processing
import pandas as pd


def set_pd_option():
    """
    Set pandas dataframe output configuration
    """
    pd.set_option('display.max_rows', 25)
    pd.set_option('display.min_rows', 25)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 2)


def feature_list() -> list:
    """
    :return: List of features of dataset
    """
    # TODO: Add each feature to separate list
    df = data_processing.get_train()
    for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
        cols = [col for col in df.columns if group in col]
        return cols


set_pd_option()

X_train = feature_list()
ydf = data_processing.get_train()
ydrop = ydf.drop(labels=[''])
# y_train =

# rfr1 = RandomForestRegressor(n_estimators=1, criterion='mse', max_depth=1, max_features=1, bootstrap=True)
# rfr1.fit(X_train, y_train)


# TODO: clustering https://sklearn.org/modules/clustering.html
