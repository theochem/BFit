from sympy import integrate, log, exp, oo
from sympy.abc import a, b, c, d, e,f, g,h, x, r
from sympy.abc import n, z
print(integrate( r**2 * exp(-z*(r**2)) * log(1 + (n) * exp((-a + b)*r)) , r))