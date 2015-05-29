import numpy as np;
from fitting.density import *
from fitting.gbasis import *
from fitting.density import *
import matplotlib.pyplot as plt
import scipy.optimize

def clenshaw_curtis(atomic_number): #remove zero
    points = np.arange(0, 200)
    points2 = np.arange(0, 300)
    quadrature =  1/(2 * atomic_number) * (1 - np.cos( np.pi * points / (200 * 2)))
    quadrature2 = 25 * (1 - np.cos(np.pi * points2 / 600))
    quadrature = np.append(quadrature, quadrature2)
    quadrature = np.append(quadrature, [50, 75, 100])
    return np.sort(quadrature)

Be_grid = clenshaw_curtis(4)
Be_grid = Be_grid.reshape((len(Be_grid), 1))


a = gbasis.UGBSBasis("be")
be_UGBS = (a.exponents("s"))
print(be_UGBS)
be = atomic_slater_density.Atomic_Density(r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater", Be_grid)
be_rho = be.atomic_density()
print(np.shape(be_rho), np.shape(Be_grid), np.shape(be_UGBS))

def func(x):
    rho, grid, UGBS = be_rho, Be_grid, be_UGBS
    #print(np.shape(rho))
    inner = x * np.exp(-2 * UGBS * grid**2) # 503 x 25
    #print(np.shape(inner))
    sum = np.sum(inner, axis=1)
    sum = np.reshape(sum, (503, 1))
    #print(np.shape(sum))
    diff = rho - sum # 503 x 503
    #print(np.shape(diff))
    #print(np.sum(diff**2, axis=1))
    return np.sum(diff**2)


"""
list = []
for x in np.linspace(0, 5, num=100):
    list.append(func(x, args))
plt.plot(list)
plt.show()"""

x0 = np.linspace(0, 1, num = 25)
c = scipy.optimize.fmin_l_bfgs_b(func, x0,  bounds=[(0, None) for x in range(0, 25)], approx_grad=True )
print(c)
coeff = c[0]

