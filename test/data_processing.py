import os
import numpy as np
import pandas as pd
import numerapi
import log
import argparse

# Initialize Numerai's API (important)
napi = numerapi.NumerAPI(verbosity="info")
# Directory to save the data to (set this yourself)
out = "C:\Python\Projects\stockmarket\data"


def main(flags):
    print(flags.dir[0])

    def download_current_data(directory: str):
        """
        Downloads the data for the current round
        :param directory: The path to the directory where the data needs to be saved
        """
        current_round = napi.get_current_round()
        full_path = f'{directory}/numerai_dataset_{napi.get_current_round()}'
        if os.path.isdir(f'{directory}/numerai_dataset_{current_round}/'):
            log.logger.info(msg=f"You already have the newest data. Current round is: {current_round}")
        else:
            napi.download_current_dataset(dest_path=directory, unzip=True)

        # Throw out files we don't need + old files
        if os.path.exists(out + '\\numerai_dataset_' + str(current_round) + '.zip'):
            os.remove(out + '\\numerai_dataset_' + str(current_round) + '.zip')
        if os.path.exists(out + '\\numerai_dataset_' + str(current_round - 1) + '\\numerai_tournament_data.csv'):
            os.remove(out + '\\numerai_dataset_' + str(current_round - 1) + '\\numerai_tournament_data.csv')
        if os.path.exists(out + '\\numerai_dataset_' + str(current_round - 1) + '\\numerai_training_data.csv'):
            os.remove(out + '\\numerai_dataset_' + str(current_round - 1) + '\\numerai_training_data.csv')
        if os.path.exists(out + '\\numerai_dataset_' + str(current_round - 1)):
            os.removedirs(out + '\\numerai_dataset_' + str(current_round - 1))
        if os.path.exists(full_path + 'analysis_and_tips.ipynb'):
            os.remove(full_path + 'analysis_and_tips.ipynb')
        if os.path.exists(full_path + 'example_model.py'):
            os.remove(full_path + 'example_model.py')
        if os.path.exists(full_path + 'example_model.r'):
            os.remove(full_path + 'example_model.r')
        if os.path.exists(full_path + 'example_predictions.csv'):
            os.remove(full_path + 'example_predictions.csv')
        else:
            log.logger.info(msg=f"Latest data downloaded successfully. Current round is: {current_round}")
            return


    def load_data(directory: str, reduce_memory: bool = True) -> tuple:
        """
        Get data for current round in pandas dataframe
        :param reduce_memory: Reduce all features to 32-bit floats
        :param directory: The path to the directory where the data needs to be saved
        :return: A tuple containing the datasets
        """
        full_path = f'{directory}/numerai_dataset_{napi.get_current_round()}/'
        train_path = full_path + 'numerai_training_data.csv'
        test_path = full_path + 'numerai_tournament_data.csv'
        train = pd.read_csv(train_path)
        log.logger.info(msg="Train data loaded.")
        test = pd.read_csv(test_path)
        log.logger.info(msg="Test data loaded.")

        if reduce_memory:
            num_features = [f for f in train.columns if f.startswith("feature")]
            # Reduce column types to float32 for memory
            train[num_features] = train[num_features].astype(np.float32)
            test[num_features] = test[num_features].astype(np.float32)

        val = test[test['data_type'] == 'validation']
        dataset = train, val, test
        return dataset


    def get_train():
        df = load_data(directory=out, reduce_memory=True)
        return df[0]


    def get_val():
        df = load_data(directory=out, reduce_memory=True)
        return df[1]


    def get_test():
        df = load_data(directory=out, reduce_memory=True)
        return df[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        dest='dir',
        nargs='+',
        help='This is the working directory'
    )
    flags, unparsed = parser.parse_known_args()
    main(flags)
