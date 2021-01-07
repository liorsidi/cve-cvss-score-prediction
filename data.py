import os
import pickle

import pandas as pd

def get_data(raw_dir):
    train_X = pd.read_csv(os.path.join(raw_dir, "train_X.csv"), index_col=False)
    train_y = pd.read_csv(os.path.join(raw_dir, "train_y.csv"), index_col=False)
    test_X = pd.read_csv(os.path.join(raw_dir, "test_X.csv"), index_col=False)
    return train_X, train_y, test_X


def save_pickle(obj, location):
    with open(location, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(location):
    with open(location, 'rb') as handle:
        return pickle.load(handle)