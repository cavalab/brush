import brushgp
import numpy as np

# input dimensions
m = 6
n = 4

X = np.ones((m, n), dtype=np.float32)
y = np.ones((m, 1), dtype=np.float32)

data = brushgp.Data(X, y)
search_space = brushgp.SearchSpace(data)

prog = brushgp.Program(search_space, 1, 0, 1)