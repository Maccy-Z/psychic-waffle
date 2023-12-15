from findiff import FinDiff
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=3)

x = np.linspace(-np.pi, np.pi, 100)
dx = x[1] - x[0]
f = np.sin(x)

d_dx = FinDiff(0, dx)
df_dx = d_dx(f)

plt.plot(f)
plt.plot(df_dx)
plt.show()