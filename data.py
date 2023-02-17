from typing import Tuple

import pandas as pd


def load_pulsar_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return a tuple containing two dataframes of pulsar prediction data, the train and test splits respectively.
    target_class holds the label (0 or 1), indicating whether the input corresponds to a pulsar.

    :return: 2-tuple of train, test data frames
    """
    return pd.read_csv(f"data/pulsar_data_train.csv"), pd.read_csv(f"data/pulsar_data_test.csv")


if __name__ == '__main__':
    train_df, _ = load_pulsar_data()
    train_df