import numpy as np;
from fitting.density import *
from fitting.gbasis import *
from fitting.density import *
import matplotlib.pyplot as plt
import scipy.optimize


def clenshaw_curtis(atomic_number, gridpoints1, gridpoints2):
    """
    does not include zero
    """
    points = np.arange(0, gridpoints1)
    points2 = np.arange(1, gridpoints2)

    quadrature =  1/ (2 * atomic_number) * (1 - np.cos( np.pi * points / (gridpoints1 * 2)))
    quadrature2 = 25 * (1 - np.cos(np.pi * points2 / (gridpoints2 * 2)))

    quadrature = np.concatenate((quadrature, quadrature2, [50, 75, 100]), axis=1)
    return np.sort(quadrature)

# Create A Clenshaw Grid Based on Beryllium that has 502 points,
Be_grid = clenshaw_curtis(4, 200, 300)
Be_grid = Be_grid.reshape((len(Be_grid), 1)) # 502, 1

#Obtain UGBS for Beryllium
a = gbasis.UGBSBasis("be")
be_UGBS = (a.exponents("s"))

#Obtain Electron Density for be from the clenshaw grid
file_path = r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater"
be = atomic_slater_density.Atomic_Density(r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater", Be_grid)
be_rho = be.atomic_density()

#Optional
#Plot the Electron Density for beryllium from the clenshaw grid
plt.plot((be_rho))
plt.show()


#############################################################################################################################
########################################## FMIN_L_BFGS_B ###################################################################
#############################################################################################################################
def total_density_cost_function(x, *args):
    """
    """
    UGBS, grid, rho = args[0], args[1], args[2]
    gaussian = x * np.exp(-2 * UGBS * grid**2) # 503 x 25
    sum_gaussians = np.sum(gaussian, axis=1)
    sum_gaussians = np.reshape(sum_gaussians, (len(sum_gaussians), 1))
    residual = rho - sum_gaussians
    sum_squared_residuals = np.sum(residual**2)
    return sum_squared_residuals

print(total_density_cost_function(0, be_UGBS, Be_grid, be_rho) == np.sum(be_rho**2 ))
# Initial Guess
x0 = np.linspace(0, 1, num = 25)
#This Is The Optimization that is Used
c = scipy.optimize.fmin_l_bfgs_b(total_density_cost_function, x0, args=(be_UGBS, Be_grid, be_rho),  bounds=[(0, None) for x in range(0, 25)], approx_grad=True )
coeff = c[0]

# Another Grid that takes small increments towards 10
grid = np.arange(0.0, 10.0, 0.0001)
def f(opt_coeffs, grid):
    grid = np.reshape(grid, (len(grid), 1))  #100000 x 1
    print(np.shape(coeff * np.exp(be_UGBS * -2 * grid**2)))    #(100000 x 25)
    return np.sum(coeff * np.exp(be_UGBS * -2 * grid**2), axis = 1)

density_approx = f(coeff, Be_grid)   #opt_coeffs refers to the optimized coefficients
# turn the density approximate into 1D array, if yours is 2D array (i.e. column array)
density_approx = np.ravel(density_approx)
#print("density approx", np.trapz(Be_grid**2 * density_approx, Be_grid) )       #number of electrons by integrating the approximate model


#############################################################################################################################
########################################## NON-NEGATIVE LEAST SQUARES #######################################################
#############################################################################################################################
"""
Performing non-negative least squares solver
where b is the electron density and A is the """
def total_density_cost_function(x, *args):

    UGBS, grid, rho = args[0], args[1], args[2]
    gaussian = np.exp(-2 * UGBS * grid**2) # 503 x 25
    return gaussian

# This uses clenshaw grid and electron density
total = total_density_cost_function(1, be_UGBS, Be_grid, be_rho)
print(np.shape(total), np.shape(be_rho))
solution = scipy.optimize.nnls(total, be_rho)[0]
print(solution, coeff)
density_approx2 = f(solution, Be_grid)
plt.plot((density_approx), "b")
plt.plot(density_approx2, "r")
plt.plot(be_rho, "g")
plt.show()
density_approx = np.reshape(density_approx, (len(density_approx), 1))
print(np.shape(be_rho), np.shape(density_approx))
print(np.absolute(np.sum(be_rho - density_approx)))
density_approx2 = np.reshape(density_approx2, (len(density_approx2), 1))
print(np.sum(be_rho - density_approx2))
print("Density approx", np.shape(density_approx2))
