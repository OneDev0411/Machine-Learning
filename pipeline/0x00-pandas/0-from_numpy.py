#!/usr/bin/env python3
"""function that creates a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """array: np.ndarray
    Returns: pd.Dataframe from array with its columns labeled in
    alphabetic order and capitalized
    """
    cap_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    df = pd.DataFrame(array, columns=[x for x in cap_letters[
        0: array.shape[1]]])
    return df
