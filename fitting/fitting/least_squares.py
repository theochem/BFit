import numpy as np;
from fitting.density import *
from fitting.gbasis import *
from fitting.density import *
import matplotlib.pyplot as plt
import scipy.optimize

def clenshaw_curtis(num_of_points, atomic_number):
    points = np.arange(0, 200)
    points2 = np.arange(0, 300)
    quadrature =  1/(2 * atomic_number) * (1- np.cos( np.pi * points / (200 * 2)))
    qudarature2 = 25 * (1 - np.cos(np.pi * points2 / 600))
    quadrature = np.append(quadrature, qudarature2)
    quadrature = np.append(quadrature, [50, 75, 100])

    return np.sort(quadrature)

Be_grid200 = (clenshaw_curtis(200, 4))
Be_grid200 = Be_grid200.reshape((len(Be_grid200), 1))
#Be_grid300 = clenshaw_curtis(300, 4)


a = gbasis.UGBSBasis("be")
be_UGBS = (a.exponents("s"))

be = atomic_slater_density.Atomic_Density(r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater", Be_grid200)
be_rho = be.atomic_density()
print(np.shape(be_rho), np.shape(Be_grid200), np.shape(be_UGBS))

def func(x):
    rho, grid, UGBS = be_rho, Be_grid200, be_UGBS
    inner = x * np.exp(-2 * UGBS * grid**2)
    #print(np.shape(inner))
    diff = rho - np.sum(inner, axis = 0)
    #print(np.shape(diff))
    #print(np.sum(diff**2))
    return np.sum(diff**2)


"""
list = []
for x in np.linspace(0, 5, num=100):
    list.append(func(x, args))
plt.plot(list)
plt.show()"""

x0=np.linspace(0, 1, num = 25)
c = (scipy.optimize.fmin_l_bfgs_b(func, x0,  bounds=[(0, None) for x in range(0, 25)], approx_grad=True ))
print(c)
plt.plot(func(c[0]))
plt.show()