#!/usr/bin/env python3
"""script that prints the location of a specific user"""
import requests
import sys
import time


if __name__ == '__main__':
    """ user is passed as first argument of the script with
    the full API URL"""
    response = requests.get(sys.argv[1])
    if response.status_code == 200:
        print(response.json()["location"])
    elif response.status_code == 403:
        x = int(time.time()) - int(response.headers['X-Ratelimit-Reset'])
        print('Reset in {} min'.format(int(abs(x) / 60)))
    else:
        print('Not found')
