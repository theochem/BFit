from fitting.density import *
from fitting.gbasis import *
from fitting.density import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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
#Be_grid = np.arange(0, 10.0, 0.1)
Be_grid = Be_grid.reshape((len(Be_grid), 1)) # 502, 1

#Obtain UGBS for Beryllium
a = gbasis.UGBSBasis("be")
be_UGBS = (a.exponents("s")); print(be_UGBS)

#Obtain Electron Density for Beryllium from the clenshaw grid
file_path = r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater"
be = atomic_slater_density.Atomic_Density(r"\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater", Be_grid)
be_rho = be.atomic_density()

#Optional
#Plot the Electron Density for beryllium from the clenshaw grid
#plt.plot(np.ravel(Be_grid), np.ravel(be_rho), 'r--')
#plt.show()


#############################################################################################################################
########################################## FMIN_L_BFGS_B ####################################################################
#############################################################################################################################
be_UGBS = np.concatenate(( [0.02*1.95819475**25, 0.02*1.95819475**26, 0.02*1.95819475**27, 0.02*1.95819475**28, 0.02*1.95819475**29,
                0.02*1.95819475**30, 0.02*1.95819475**31], be_UGBS))
length_UGBS = np.shape(be_UGBS)[0]
""", 0.02*1.95819475**32, 0.02*1.95819475**33, 0.02*1.95819475**34, 0.02*1.95819475**35
                ,0.02*1.95819475**36, 0.02*1.95819475**37,0.02*1.95819475**38, 0.02*1.95819475**39, 0.02*1.95819475**40, 0.02*1.95819475**41
                , 0.02*1.95819475**42, 0.02*1.95819475**43, 0.02*1.95819475**44, 0.02*1.95819475**45, 0.02*1.95819475**46, 0.02*1.95819475**47,
                0.02*1.95819475**48, 0.02*1.95819475**49, 0.02*1.95819475**50"""
def total_density_cost_function(x, *args):
    """
    """
    UGBS, grid, rho = args[0], args[1], args[2]
    gaussian = x * np.exp(np.longdouble(-1.9999999999 * UGBS * grid**2)) # 503 x 25
    sum_gaussians = np.sum(gaussian, axis=1)
    sum_gaussians = np.reshape(sum_gaussians, (len(sum_gaussians), 1))
    residual = rho - sum_gaussians
    sum_squared_residuals = np.sum(residual**2)
    return sum_squared_residuals
print(total_density_cost_function(0, be_UGBS, Be_grid, be_rho) == np.sum(be_rho**2 ))

# Initial Guess
x0 = np.linspace(0, 1, num = length_UGBS)
#This Is The Optimization that is Used
optimized = scipy.optimize.fmin_l_bfgs_b(total_density_cost_function, x0, args=(be_UGBS, Be_grid, be_rho),  bounds=[(0, None) for x in range(0, length_UGBS)], approx_grad=True )
coeff = optimized[0]

# Another Grid that takes small increments towards 10
grid = np.arange(0.0, 10.0, 0.0001)
def f(opt_coeffs, grid):
    grid = np.reshape(grid, (len(grid), 1))  #100000 x 1
    print(np.shape(coeff * np.exp(be_UGBS * -2 * grid**2)))    #(100000 x 25)
    return np.sum(coeff * np.exp(be_UGBS * -2 * grid**2), axis = 1)

density_approx = f(coeff, Be_grid)   #opt_coeffs refers to the optimized coefficients
# turn the density approximate into 1D array, if yours is 2D array (i.e. column array)
density_approx = np.ravel(density_approx)
density_resized = np.reshape(density_approx, (len(density_approx), 1))
print(np.shape(Be_grid), np.shape(density_resized))


#############################################################################################################################
########################################## NON-NEGATIVE LEAST SQUARES #######################################################
#############################################################################################################################
"""
Performing non-negative least squares solver
where b is the electron density and A is the """
def cofactor_matrix(UGBS, grid):
    inner = UGBS * grid**2 # 503 x 25
    gaussian = np.exp(-2 * inner) # 503 x 25
    return gaussian

# This uses clenshaw grid and electron density
cofactor_mat = cofactor_matrix(be_UGBS, Be_grid)
print("The Shape of A/Cofactor Matrix is ", np.shape(cofactor_mat))
NNLS_coeff = scipy.optimize.nnls(cofactor_mat, be_rho)[0]
print("These Are The Coefficients Obtained From f_min_BFGS\n", coeff, "\n")
print("These are the Coefficients Obtained from NNLS\n", NNLS_coeff, "\n")
print( np.shape(NNLS_coeff), np.shape(Be_grid))

def plot_function(coefficients, UGBS, grid):
    inner = UGBS * grid**2
    gaussian = np.dot(np.exp(-2 * inner), NNLS_coeff)
    return gaussian

NNLS_density_approx = plot_function(NNLS_coeff, NNLS_coeff, np.reshape(Be_grid, (len(Be_grid), 1)))
print("Shape", np.shape(NNLS_density_approx))
density_approx = np.reshape(density_approx, (len(density_approx), 1)) # this is for the plot

#Plotting
log_grid = np.log(np.ravel(Be_grid))
grid = np.ravel(Be_grid)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
ax.plot(grid, (density_approx), "b", label="F_MIN_BFGS_B")
ax.plot(grid, (NNLS_density_approx), "r--", label="NNLS")
ax.plot(grid, (be_rho), "g", label="True Density ")
ax.legend(bbox_to_anchor=(1.00, 1), loc=2, borderaxespad=0.)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(1.2, (0.1 + 0.75)/2, "T - F_Min_BFGS_B " + str(np.round(np.asscalar(np.absolute(np.sum(be_rho - density_approx))), decimals=2))
                            + "\n T - NNLS " + str(np.round(np.asscalar(np.absolute(np.sum(be_rho - NNLS_density_approx))), decimals=2))
                            + "\n F_Min - NNLS " + str(np.round(np.asscalar(np.absolute(np.sum(density_approx - NNLS_density_approx))), decimals=2)),
        transform=ax.transAxes, fontsize=14,
        verticalalignment='center', horizontalalignment='center', bbox=props)

plt.show()

#Integration
print("Integration of Beryllium for NNLS Using Clenshaw Grid is ", np.trapz(np.ravel(Be_grid**2) * np.ravel(NNLS_density_approx),
                                                                            np.ravel(Be_grid)))
print("Integration Of Beryllium for f_Min_LGBS Using Clenshaw Grid is ", np.trapz(np.ravel(Be_grid**2 * density_resized), np.ravel(Be_grid)) )       #number of electrons by integrating the approximate model

#Taking The Difference Between the Techniques
print("The Difference between True and F_Min_BFGS_B ", np.absolute(np.sum(be_rho - density_approx)))
NNLS_density_approx = np.reshape(NNLS_density_approx, (len(NNLS_density_approx), 1))
print("The Difference between True and NNLS ", np.absolute(np.sum(be_rho - NNLS_density_approx)))
print("The Difference between F_Min_BFGS_B and NNLS ", np.sum(density_approx - NNLS_density_approx))
