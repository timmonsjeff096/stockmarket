import os
import numpy as np
import random as rn
import pandas as pd
import numerapi

# Initialize Numerai's API (important)
NAPI = numerapi.NumerAPI(verbosity="info")

# Directory to save the data to (set this yourself)
DIR = "C:\Python\Projects\stockmarket\data"

# Set seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Surpress Pandas warnings
pd.set_option('chained_assignment', None)


def download_current_data(directory: str):
    """
    Downloads the data for the current round
    :param directory: The path to the directory where the data needs to be saved
    """
    current_round = NAPI.get_current_round()
    full_path = f'{directory}/numerai_dataset_{NAPI.get_current_round()}/'
    if os.path.isdir(f'{directory}/numerai_dataset_{current_round}/'):
        print(f"You already have the newest data. Current round is: {current_round}")
    else:
        NAPI.download_current_dataset(dest_path=directory, unzip=True)
    # Throw out files we don't need
    if os.path.exists(DIR + '/numerai_dataset_' + str(current_round) + '.zip'):
        os.remove(DIR + '/numerai_dataset_' + str(current_round) + '.zip')
    if os.path.exists(full_path + 'analysis_and_tips.ipynb'):
        os.remove(full_path + 'analysis_and_tips.ipynb')
    if os.path.exists(full_path + 'example_model.py'):
        os.remove(full_path + 'example_model.py')
    if os.path.exists(full_path + 'example_model.r'):
        os.remove(full_path + 'example_model.r')
    if os.path.exists(full_path + 'example_predictions.csv'):
        os.remove(full_path + 'example_predictions.csv')
    else:
        print(f"Latest data downloaded. Current round is: {current_round}")


def load_data(directory: str, reduce_memory: bool = True) -> tuple:
    """
    Get data for current round
    :param reduce_memory: Reduce all features to 32-bit floats
    :param directory: The path to the directory where the data needs to be saved
    :return: A tuple containing the datasets
    """

    print('Loading the data')
    full_path = f'{directory}/numerai_dataset_{NAPI.get_current_round()}/'
    train_path = full_path + 'numerai_training_data.csv'
    test_path = full_path + 'numerai_tournament_data.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if reduce_memory:
        num_features = [f for f in train.columns if f.startswith("feature")]
        # Reduce column types to float32 for memory
        train[num_features] = train[num_features].astype(np.float32)
        test[num_features] = test[num_features].astype(np.float32)

    val = test[test['data_type'] == 'validation']
    dataset = train, val, test
    return dataset


def feature_list(df: pd.DataFrame) -> list:
    """
    :return: List of features of dataset
    """
    df = get_training_data()
    for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
        cols = [col for col in df.columns if group in col]
        df[f"feature_{group}"] = df[cols].index
        return cols


def get_training_data() -> pd.DataFrame:
    """
    :return: pandas dataframe with training data
    """
    df = data_set[0]
    return df


def get_test_data() -> pd.DataFrame:
    """
    :return: pandas dataframe with test data
    """
    df = data_set[2]
    return df


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

# ######################################################################################################################


download_current_data(DIR)
data_set = load_data(DIR, reduce_memory=True)
