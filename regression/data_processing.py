import os
from logging.handlers import RotatingFileHandler
from logging import Formatter
import numpy as np
import pandas as pd
import numerapi
import json
import logging
from sklearn import linear_model

napi = numerapi.NumerAPI(verbosity="info")


def setup_config() -> dict:
    with open("config.json") as cfg:
        myjson = json.load(cfg)
        cfg_values = {"output_directory": myjson.get("output_directory"),
                      "data_directory": myjson.get("data_directory"),
                      "log_directory": myjson.get("log_directory")}
        json.dumps(cfg_values)
        return cfg_values


def setup_logging(log_dir):

    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"
    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


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
    """
    # Initialize Numerai's API (important)
    napi = numerapi.NumerAPI(verbosity="info")
    current_round = napi.get_current_round()
    last_round = current_round - 1
    full_path = f'{data_dir}/numerai_dataset_{current_round}'

    if os.path.isdir(full_path):
        logging.info(f"Latest data already downloaded. Current round is {current_round}")
    else:
        logging.info(f"Downloading data for round {current_round}")
        napi.download_current_dataset(data_dir, unzip=True)

    # Throw out files we don't need + old files
    if os.path.exists(full_path + '.zip'):
        os.remove(full_path + '.zip')
    if os.path.exists(data_dir + '/numerai_dataset_' + str(last_round) + '/numerai_tournament_data.csv'):
        os.remove(data_dir + '/numerai_dataset_' + str(last_round) + '/numerai_tournament_data.csv')
    if os.path.exists(data_dir + '/numerai_dataset_' + str(last_round) + '/numerai_training_data.csv'):
        os.remove(data_dir + '/numerai_dataset_' + str(last_round) + '/numerai_training_data.csv')
    if os.path.exists(data_dir + '/numerai_dataset_' + str(last_round)):
        os.removedirs(data_dir + '/numerai_dataset_' + str(last_round))
    if os.path.exists(full_path + '/analysis_and_tips.ipynb'):
        os.remove(full_path + '/analysis_and_tips.ipynb')
    if os.path.exists(full_path + '/example_model.py'):
        os.remove(full_path + '/example_model.py')
    if os.path.exists(full_path + '/example_model.r'):
        os.remove(full_path + '/example_model.r')
    if os.path.exists(full_path + '/example_predictions.csv'):
        os.remove(full_path + '/example_predictions.csv')
    else:
        return


def load_data(data_dir, reduce_memory: bool = True) -> tuple:
    """
    Get data for current round in pandas dataframe
    :return: A tuple containing the datasets
    """
    # Initialize Numerai's API (important)
    napi = numerapi.NumerAPI(verbosity="info")
    full_path = f'{data_dir}/numerai_dataset_{napi.get_current_round()}/'
    train_path = full_path + 'numerai_training_data.csv'
    test_path = full_path + 'numerai_tournament_data.csv'
    logging.info("Loading train data")
    train = pd.read_csv(train_path)
    logging.info("Loading test data")
    test = pd.read_csv(test_path)

    if reduce_memory:
        num_features = [f for f in train.columns if f.startswith("feature")]
        # Reduce column types to float32 for memory
        train[num_features] = train[num_features].astype(np.float32)
        test[num_features] = test[num_features].astype(np.float32)

    val = test[test['data_type'] == 'validation']
    dataset = train, val, test
    return dataset


def get_train(data_dir) -> pd.DataFrame:
    df = load_data(data_dir, reduce_memory=True)
    return df[0]


def get_val(data_dir) -> pd.DataFrame:
    df = load_data(data_dir, reduce_memory=True)
    return df[1]


def get_test(data_dir) -> pd.DataFrame:
    df = load_data(data_dir, reduce_memory=True)
    return df[2]


conf = setup_config()
setup_logging(conf.get("log_directory"))
data_path = conf.get("data_directory")
out_path = conf.get("output_directory")

download_current_data(data_path)
train_data = get_train(data_path)
test_data = get_test(data_path)
