from sympy import integrate, log, exp, oo
from sympy.abc import a, b, c, d, e,f, g,h, x, r

print(integrate( r**2 * exp(-a*r**2) * log(c*exp(-b*r))  , (r,0, oo)))