import numpy as np

p, w = np.polynomial.legendre.leggauss(100)

print(p[0:3])
p1 = -0.99971373
p2 = -0.99849195
p3 = 0.01562898

exponent = 12.683501
r = p1
slateroneS = 2 * (exponent)**3/2 * np.exp(-exponent * r)
print(slateroneS)
"""
points one

p1 = 29010259.76

be = 20 has 4 electrons  0.2
he = 11 has 2 electrons  0.1818
li = 15 has 3 electrons  0.2
"""

