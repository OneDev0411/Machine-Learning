#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(
    student_grades,
    bins=[
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100],
    edgecolor='black')
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xticks(np.arange(0, 101, 10))
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title("Project A")
plt.show()
