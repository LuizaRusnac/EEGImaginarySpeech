import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

daub = signal.daub(4)
x, phi, psi = signal.cascade(daub, 10)

# print(daub)
# plt.figure()
# plt.plot(daub)
# plt.show()
# plt.close()
plt.figure()
plt.plot(psi)
plt.show()
plt.close()