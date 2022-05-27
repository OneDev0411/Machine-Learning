#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)
df["Close"] = df["Close"].fillna(method='ffill')
df["High"] = df["High"].fillna(df['Close'].shift(1))
df["Low"] = df["Low"].fillna(df['Close'].shift(1))
df["Open"] = df["Open"].fillna(df['Close'].shift(1))

print(df.head())
print(df.tail())