import numpy as np
import preprocessing

m = np.array([[[1, 1, 4], [1, 1, 3], [1, 1, 7]],[[0, 0, 6], [0, 0, 2], [0, 0, 3]],[[2, 2, 3], [2, 2, 4], [2, 2, 5]]])
print(m)
mnorm = preprocessing.featureStd(m)
print(mnorm)