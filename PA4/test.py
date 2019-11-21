import numpy as np
x = [[1,2], [2,3], [3,4], [2,5], [7,8], [9,0]]
a = [[1,2], [3,4], [5,6]]
d = np.expand_dims(a, axis=1)
print(x - d)