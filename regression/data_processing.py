import os
from logging.handlers import RotatingFileHandler
from logging import Formatter
import numpy as np
import pandas as pd
import numerapi
import json
import logging


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
            if len(myjson["base_directory"]) or len(myjson["output_directory"]) or len(
                    myjson["data_directory"]) or len(myjson["log_directory"]) != 0:
                cfg_values = {"base_directory": myjson.get("base_directory"),
                              "output_directory": myjson.get("output_directory"),
                              "data_directory": myjson.get("data_directory"),
                              "log_directory": myjson.get("log_directory")}
                json.dumps(cfg_values)
                return cfg_values
            else:
                cfg_values = {"base_directory": "",
                              "output_directory": "",
                              "data_directory": "",
                              "log_directory": ""}
                json.dumps(cfg_values)
                logging.error("Enter directory information in configuration file")
                exit(101)
    else:
        with open("config.json", "w") as jsonFile:
            cfg_values = {"base_directory": "", "output_directory": "", "data_directory": "", "log_directory": ""}
            json.dump(cfg_values, jsonFile)
            logging.error("Enter directory information in configuration file")
            exit(101)


class DataProcessing:
    def __init__(self):
        cfg_values = setup_config()
        self.base_dir = cfg_values["base_directory"]
        self.log_dir = self.base_dir + cfg_values['log_directory']
        initialize_logger(self.log_dir)
        self.data_dir = self.base_dir + cfg_values['data_directory']
        self.out_dir = self.base_dir + cfg_values['output_directory']
        self.napi = numerapi.NumerAPI(verbosity="info")
        self.current_round = self.napi.get_current_round()

    def download_current_data(self):
        """
        Download latest numerai data from numerapi
        """
        data_dir = self.data_dir
        current_round = self.current_round
        napi = self.napi
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
        self.download_current_data()
        data_dir = self.data_dir
        full_path = f'{data_dir}/numerai_dataset_{self.current_round}/'
        training_path = full_path + 'numerai_training_data.csv'
        tournament_path = full_path + 'numerai_tournament_data.csv'
        logging.info("Loading training data")
        training = pd.read_csv(training_path, header=0)
        logging.info("Loading tournament data")
        tournament = pd.read_csv(tournament_path, header=0)

        if reduce_memory:
            num_features = [f for f in training.columns if f.startswith("feature")]
            # Reduce column types to float32 for memory
            training[num_features] = training[num_features].astype(np.float32)
            tournament[num_features] = tournament[num_features].astype(np.float32)

        dataset = training, tournament
        logging.info("Data loaded successfully")
        return dataset

    def get_data(self) -> tuple:
        """
        Call for getting training and tournament dataframes(tuple)
        :return:
        """
        dataframe = self.load_data(reduce_memory=True)
        return dataframe
