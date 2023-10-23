import numpy as np


M = 2

A = np.array([[(j - 0.5)**(2 * i - 1) for j in range(1, M + 1)] for i in range(1, M + 1)])

b = np.zeros(M)
b[0] = 0.5

coef = np.linalg.solve(A, b)

print(A)

print(coef)
print(27. / 24, -1. /24)