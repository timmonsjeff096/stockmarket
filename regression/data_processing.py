import os
from logging.handlers import RotatingFileHandler
from logging import Formatter
import numpy as np
import pandas as pd
import numerapi
import json
import logging

napi = numerapi.NumerAPI(verbosity="info")


def initialize_logger(log_dir):
    """
    # Enable logging functionality
    """
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    main_logger = logging.getLogger("")
    main_logger.setLevel(logging.DEBUG)

    if main_logger.hasHandlers():
        main_logger.handlers.clear()

    console_handler = logging.StreamHandler(stream=None)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_console_format))

    file_handler = RotatingFileHandler('{}\logger.log'.format(log_dir), maxBytes=10 ** 6, backupCount=3)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)


def setup_config() -> dict:
    if os.path.isfile("config.json"):
        with open("config.json") as cfg:
            myjson = json.load(cfg)
            if len(myjson["output_directory"]) and len(myjson["data_directory"]) and len(
                    myjson["log_directory"]) == 0:
                cfg_values = {"output_directory": "",
                              "data_directory": "",
                              "log_directory": ""}
                json.dumps(cfg_values)
                logging.error("Enter directory information in configuration file")
                exit(101)
            else:
                cfg_values = {"output_directory": myjson.get("output_directory"),
                              "data_directory": myjson.get("data_directory"),
                              "log_directory": myjson.get("log_directory")}
                json.dumps(cfg_values)
                return cfg_values
    else:
        with open("config.json", "w") as jsonFile:
            cfg_values = {"output_directory": "", "data_directory": "", "log_directory": ""}
            json.dump(cfg_values, jsonFile)
            logging.error("Enter directory information in configuration file")
            exit(101)


class DataProcessing:
    def __init__(self):
        cfg_values = setup_config()
        initialize_logger(cfg_values['log_directory'])
        self.data_dir = cfg_values['data_directory']
        self.out_dir = cfg_values['output_directory']

    def download_current_data(self):
        """
        Download latest numerai data from numerapi
        """
        data_dir = self.data_dir
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

    def load_data(self, reduce_memory: bool = True) -> tuple:
        """
        Get data for current round in pandas dataframe
        :return: A tuple containing the datasets
        """
        # Initialize Numerai's API (important)
        data_dir = self.data_dir
        full_path = f'{data_dir}/numerai_dataset_{napi.get_current_round()}/'
        train_path = full_path + 'numerai_training_data.csv'
        test_path = full_path + 'numerai_tournament_data.csv'
        logging.info("Loading train data")
        train = pd.read_csv(train_path, header=0)
        logging.info("Loading test data")
        test = pd.read_csv(test_path, header=0)

        if reduce_memory:
            num_features = [f for f in train.columns if f.startswith("feature")]
            # Reduce column types to float32 for memory
            train[num_features] = train[num_features].astype(np.float32)
            test[num_features] = test[num_features].astype(np.float32)

        val = test[test['data_type'] == 'validation']
        dataset = train, val, test
        logging.info("Data loaded successfully")
        return dataset

    def get_data(self) -> tuple:
        """
        Call for getting train, val, test(tuple)
        :return:
        """
        dataframe = self.load_data(reduce_memory=True)
        return dataframe


########################################################################################################################
data_processing = DataProcessing()
data_tup = data_processing.get_data()
train_data = data_tup[0]
