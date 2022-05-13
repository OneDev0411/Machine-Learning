#!/usr/bin/env python3
"""function that creates a pd.DataFrame from a np.ndarray"""
import string
import pandas as pd


def from_numpy(array):
    """array: np.ndarray
    Returns: pd.Dataframe from array with its columns labeled in
    alphabetic order and capitalized
    """
    df = pd.DataFrame(array, columns=[x for x in string.ascii_uppercase[
        0: array.shape[1]]])
    return df
