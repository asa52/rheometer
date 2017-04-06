"""Generic plotting code, as Bash on Windows cannot make displayable plots."""

import numpy as np
import matplotlib.pyplot as plt

results = np.loadtxt('test.txt')

plt.plot(results[:, 0], results[:, 1])
plt.show()