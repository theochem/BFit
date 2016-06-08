
import sympy as sp
from sympy.abc import c, a, x, y, z, d,  b
from sympy import oo

# One Dimension from -inf
print(sp.integrate(c * sp.exp(-a * ((x - b)**2 + (y-d)**2)), (y, -oo, z)  ))