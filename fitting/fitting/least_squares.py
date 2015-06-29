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
file_path = r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater"
be = atomic_slater_density.Atomic_Density(r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater", Be_grid)
be_rho = be.atomic_density()


def func(x):
    rho, grid, UGBS = be_rho, Be_grid, be_UGBS
    inner = x * np.exp(-2 * UGBS * grid**2) # 503 x 25
    sum = np.sum(inner, axis=1)
    sum = np.reshape(sum, (503, 1))
    diff = rho - sum # 503 x 503
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


grid = np.arange(0.0, 10.0, 0.0001)
print(np.shape(coeff), np.shape(be_UGBS), np.shape(grid))

def f(opt_coeffs, grid):
    grid = np.reshape(grid, (len(grid), 1))  #100000 x 1
    print(np.shape(coeff * np.exp(be_UGBS * -2 * grid**2)))    #(100000 x 25)
    return np.sum(coeff * np.exp(be_UGBS * -2 * grid**2), axis = 1) #(100000 x 1)
print(coeff)
coeff = np.array([  5.46216373e+00,   0.00000000e+00,   0.00000000e+00,
         3.88524190e+00,   4.84911435e+00,   5.04914030e+00,
         7.77072403e+00,   1.14552666e+01,   1.54129327e+01,
         2.13043839e+01,   2.96460347e+01,   3.97163402e+01,
         5.22680484e+01,   6.46277459e+01,   6.95888197e+01,
         6.10717675e+01,   3.96714102e+01,   1.16335009e+01,
         1.39023514e-01,   9.61228602e-02,   4.43567089e-04,
         2.14737164e-01,   1.96237121e-01,   1.63525742e-02,
         0.00000000e+00])
density_approx = f(coeff, grid)   #opt_coeffs refers to the optimized coefficients
# turn the density approximate into 1D array, if yours is 2D array (i.e. column array)
density_approx = np.ravel(density_approx)
print(np.trapz(grid**2 * density_approx, grid) )       #number of electrons by integrating the approximate model

be = Atomic_Density(file_path, grid.reshape(len(grid), 1))
density = np.ravel(be.atomic_density())
print(np.trapz(np.power(grid, 2) * density, grid)  )                  #number of electrons by integrating the true density
import sys
print(sys.version)