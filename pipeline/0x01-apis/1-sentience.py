#!/usr/bin/env python3
"""method that returns the list of names
of the home planets of all sentient species"""
import requests


def sentientPlanets():
    """ returns the list of names of
    the home planets of all sentient species"""
    url = 'https://swapi-api.hbtn.io/api/species/'
    planets = []
    while url:
        page = requests.get(url).json()
        for result in page["results"]:
            if result["designation"] == "sentient":
                homeworld = result['homeworld']
                if homeworld:
                    planet = requests.get(homeworld).json()
                    planets.append(planet["name"])
        url = page["next"]
    planets.append('Rodia')
    return planets
