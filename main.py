import numpy as np
from matplotlib import pyplot as plt

def findiff(us):
    diff = np.zeros_like(us)

    for i in range(13):
        diff[i+1] = us[i+2] - us[i]

    return diff

def plot(us):
    plt.plot(us)
    plt.scatter(5, 0, color='red')
    plt.show()


xs = np.arange(15)

us = - xs + 6
us[:6] = 1

d_us = findiff(us)
plot(d_us)

a = -1
gps = np.array([0, 0, 0, 0, 0, 0, a, -2, a-2, -4, a-4, -6, a-6, -8, a-8])

d_gps = findiff(gps)
plot(d_gps)
d_us_new = d_us - d_gps

plot(d_us_new)

