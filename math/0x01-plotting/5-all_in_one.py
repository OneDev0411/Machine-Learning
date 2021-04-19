#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(2, 3)
fig.suptitle('All in One')
ax1.plot(y0, 'r-')
ax1.set_xlim([0, 10])

ax2.plot(x1, y1, 'm .')
ax2.xlabel('Height (in)')
ax2.ylabel('Weight (lbs)')
ax2.title("Men's Height vs Weight")
ax2.show()

ax3.plot()


ax4.plot()
ax5.plot()
