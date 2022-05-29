#!/usr/bin/env python3
"""script that displays the upcoming launch
By using the (unofficial) SpaceX API"""
import requests
import time


if __name__ == '__main__':
    """ Format: <launch name> (<date>)
    <rocket name> - <launchpad name> (<launchpad locality>) """
    url1 = 'https://api.spacexdata.com/v4/launches/upcoming'
    url2 = 'https://api.spacexdata.com/v4/rockets/'
    url3 = 'https://api.spacexdata.com/v4/launchpads/'
    resp = requests.get(url1)
    t = resp.json()[0]['date_unix']
    now = time.time()
    min = abs(t - now)
    upcoming = resp.json()[0]
    for i in resp.json():
        if i['date_unix'] < min:
            min = i['date_unix']
            upcoming = i
    rocket = requests.get(url2 + upcoming['rocket'])
    rocket = rocket.json()['name']
    launchpad = requests.get(url3 + upcoming['launchpad'])
    launchpad = launchpad.json()
    locality = launchpad['locality']
    launchpad = launchpad['name']
    print('{} ({}) {} - {} ({})'.format(
        upcoming['name'], upcoming['date_local'], rocket, launchpad, locality))
