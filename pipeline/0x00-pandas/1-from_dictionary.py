#!/usr/bin/env python3
""" python script that created a pd.DataFrame from a dictionary"""
import pandas as pd
dict = {'First':[0.0, 0.5, 1.0, 1.5],
'Second':['one', 'two', 'three', 'four']}
df = pd.DataFrame(dict, index=['A', 'B', 'C', 'D'])
