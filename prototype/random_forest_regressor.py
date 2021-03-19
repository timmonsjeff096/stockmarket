from sklearn.ensemble import RandomForestRegressor
import data_processing

data_processing.set_pd_option()
df_train = data_processing.get_training_data()
df_test = data_processing.get_test_data()
x_train = df_train.loc[:, :]
y_train = df_train.columns
x_test = df_test.loc[:, :]
y_test = df_test.columns
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

# TODO: clustering https://sklearn.org/modules/clustering.html
