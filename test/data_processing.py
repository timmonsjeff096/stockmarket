import os
import numpy as np
import pandas as pd
import numerapi
import json
import logging


def setup_config() -> dict:
    with open("config.json") as cfg:
        myjson = json.load(cfg)
        cfg_values = {"output_directory": myjson.get("output_directory"),
                      "data_directory": myjson.get("data_directory"),
                      "log_directory": myjson.get("log_directory")}
        json.dumps(cfg_values)
        return cfg_values


def setup_logging(log_dir):
    logger = logging.getLogger('sgd_svrg')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(log_dir + '\stocklog.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


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


def download_current_data(data_dir):
    """
    Download latest numerai data from numerapi
    :return: self
    """
    # Initialize Numerai's API (important)
    napi = numerapi.NumerAPI(verbosity="info")
    current_round = napi.get_current_round()
    full_path = f'{data_dir}/numerai_dataset_{napi.get_current_round()}'
    if os.path.isdir(full_path):
        # df_logger(msg=f"You already have the newest data. Current round is: {current_round}")
        print()
    else:
        napi.download_current_dataset(dest_path=data_dir, unzip=True)

    # Throw out files we don't need + old files
    if os.path.exists(full_path + '.zip'):
        os.remove(full_path + '.zip')
    if os.path.exists(data_dir + 'numerai_dataset_' + str(current_round - 1) +
                      'numerai_tournament_data.csv'):
        os.remove(data_dir + 'numerai_dataset_' + str(current_round - 1) + 'numerai_tournament_data.csv')
    if os.path.exists(data_dir + 'numerai_dataset_' + str(current_round - 1) + 'numerai_training_data.csv'):
        os.remove(data_dir + 'numerai_dataset_' + str(current_round - 1) + 'numerai_training_data.csv')
    if os.path.exists(data_dir + 'numerai_dataset_' + str(current_round - 1)):
        os.removedirs(data_dir + 'numerai_dataset_' + str(current_round - 1))
    if os.path.exists(full_path + 'dataanalysis_and_tips.ipynb'):
        os.remove(full_path + 'dataanalysis_and_tips.ipynb')
    if os.path.exists(full_path + 'example_model.py'):
        os.remove(full_path + 'example_model.py')
    if os.path.exists(full_path + 'example_model.r'):
        os.remove(full_path + 'example_model.r')
    if os.path.exists(full_path + 'example_predictions.csv'):
        os.remove(full_path + 'example_predictions.csv')
    else:
        # df_logger(msg=f"Latest data downloaded successfully. Current round is: {current_round}")
        return


def load_data(data_dir, reduce_memory: bool = True) -> tuple:
    """
    Get data for current round in pandas dataframe
    :return: A tuple containing the datasets
    """
    # Initialize Numerai's API (important)
    napi = numerapi.NumerAPI(verbosity="info")
    full_path = f'{data_dir}numerai_dataset_{napi.get_current_round()}/'
    train_path = full_path + 'numerai_training_data.csv'
    test_path = full_path + 'numerai_tournament_data.csv'
    train = pd.read_csv(train_path)
    # df_logger(msg="Train data loaded.")
    test = pd.read_csv(test_path)
    # df_logger(msg="Test data loaded.")

    if reduce_memory:
        num_features = [f for f in train.columns if f.startswith("feature")]
        # Reduce column types to float32 for memory
        train[num_features] = train[num_features].astype(np.float32)
        test[num_features] = test[num_features].astype(np.float32)

    val = test[test['data_type'] == 'validation']
    dataset = train, val, test
    return dataset


def feature_list() -> list:
    """
    :return: List of features of dataset
    """
    fl = get_train(data_dir=setup_config())
    for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:
        cols = [col for col in fl.columns if group in col]
        return cols


def get_train(data_dir):
    df = load_data(data_dir, reduce_memory=True)
    return df[0]


def get_val(data_dir):
    df = load_data(data_dir, reduce_memory=True)
    return df[1]


def get_test(data_dir):
    df = load_data(data_dir, reduce_memory=True)
    return df[2]


conf = setup_config()
setup_logging(log_dir=conf.get("log_directory"))
data = conf.get("data_directory")
out = conf.get("output_directory")



