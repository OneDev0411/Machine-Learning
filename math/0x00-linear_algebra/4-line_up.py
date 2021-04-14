#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    if len(arr1) == len(arr2):
        res = [[arr1[i]+arr2[i] for i in range(len(arr1))]]
        return res
    else:
        return None
