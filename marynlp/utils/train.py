import pandas as pd
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def split_train_val_test(text_list: list, train_test_val_ratio: float, test_val_ratio: float):
    train_split_count = int(train_test_val_ratio * len(text_list))
    train_text_set, test_val_text_set = text_list[:train_split_count], text_list[train_split_count:]

    test_split_count = int(test_val_ratio * len(test_val_text_set))
    test_text_set, val_text_set = test_val_text_set[:test_split_count], test_val_text_set[test_split_count:] 

    return train_text_set, val_text_set, test_text_set


def pd_split_train_val_test(text_list: pd.DataFrame, train_test_val_ratio: float, test_val_ratio: float):

    train_split_count = int(train_test_val_ratio * len(text_list))
    train_text_set, test_val_text_set = text_list.iloc[:train_split_count], text_list.iloc[train_split_count:]

    test_split_count = int(test_val_ratio * len(test_val_text_set))
    test_text_set, val_text_set = test_val_text_set.iloc[:test_split_count], test_val_text_set.iloc[test_split_count:] 

    return train_text_set, val_text_set, test_text_set


def split_train(n_splits: int, train_text_set: list):
    ll = len(train_text_set)
    sectors = list(range(ll // n_splits, ll - n_splits + 2, ll // n_splits))

    for i, x in enumerate(sectors):
        if i == 0:
            # first item
            yield (train_text_set[:x])
            continue
        
        yield (train_text_set[sectors[i - 1]:x])

    # for the last one
    yield (train_text_set[sectors[-1]:])
