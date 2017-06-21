import numpy as np
import matplotlib.pyplot as plt

f = np.array([np.inf, 9.33, 8.37, 7.96, 7.02, 5.87, 4.33, 3.73, 2.12, 1.17])
v = np.array([0, 1.73, 1.63, 1.44, 1.37, 1.04, 0.89, 0.57, 0.33, 0.21])

v = np.cumsum(v)
v[0] = np.inf

plt.plot(np.linspace(0, 9, 10), f, "bo--", mec="b")
plt.plot(np.linspace(0, 9, 10), v, "ro--", mec="r")
plt.xlabel("user index")
plt.show()
