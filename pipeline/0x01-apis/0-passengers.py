#!/usr/bin/env python3
""" method that returns the list of ships that
can hold a given number of passengers"""
import requests


def availableShips(passengerCount):
    """If no ship available, returns an empty list"""
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []
    while url:
        page = requests.get(url).json()
        for result in page["results"]:
            passengers = result['passengers'].replace(',', '')
            if passengers != 'n/a' and passengers != 'unknown':
                if int(passengers) >= passengerCount:
                    ships.append(result["name"])
        url = page["next"]
    return ships
