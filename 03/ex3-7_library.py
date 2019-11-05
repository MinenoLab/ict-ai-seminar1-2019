import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)

y = 2 * (x**2)

plt.plot(x, y)
plt.show()