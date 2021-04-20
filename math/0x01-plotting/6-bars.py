#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
people = ['Farrah', 'Fred', 'Felicia']
fig, ax1 = plt.subplots()
ax1.bar(people, fruit[0], 0.5, label='Apple', color='red')
ax1.bar(
    people,
    fruit[1],
    0.5,
    bottom=fruit[0],
    label='Bananas',
    color='yellow')
ax1.bar(
    people,
    fruit[2],
    0.5,
    bottom=fruit[1] +
    fruit[0],
    label='Oranges',
    color='#ff8000')
ax1.bar(
    people,
    fruit[3],
    0.5,
    bottom=fruit[2] +
    fruit[1] +
    fruit[0],
    label='Peaches',
    color='#ffe5b4')
ax1.set_title("Number of Fruit per Person")
ax1.legend()
plt.yticks(np.arange(0, 81, 10))
plt.ylabel('Quantity of Fruit')
plt.show()
