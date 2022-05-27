#!/usr/bin/env python3
"""function that loads data from a file as a pd.DataFrame"""
import pandas as pd


def from_file(filename, delimiter):
    """filename: file to load from
    delimiter: column separator
    Returns: the loaded pd.DataFrame
    """
    df = pd.read_csv(filepath_or_buffer=filename,
    delimiter=delimiter)
    return df
