# from sklearn.ensemble import RandomForestRegressor
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


set_pd_option()
df_train = data_processing.get_training_data()
df_test = data_processing.get_test_data()

for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
    cols = [col for col in df_train.columns if group in col]
    y_train = df_train.pop(cols).values
    print(y_train)

# regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# TODO: Fix this
# TODO: clustering https://sklearn.org/modules/clustering.html
