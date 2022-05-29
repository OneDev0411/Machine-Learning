#!/usr/bin/env python3
"""script that displays the number of launches per rocket"""
from urllib import response
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    responses = requests.get(url).json()
    dic = {}
    for response in responses:
        rocket = response['rocket']
        rurl = "https://api.spacexdata.com/v4/rockets/"
        rrocket = requests.get(rurl + rocket).json()
        rocket_name = rrocket["name"]
        if rocket_name in dic.keys():
            dic[rocket_name] = dic[rocket_name] + 1
        else:
            dic[rocket_name] = 1
    for key, value in reversed(sorted(dic.items(),
                               key=lambda key: key[1])):
        print("{}: {}".format(key, value))
