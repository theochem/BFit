from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import numpy as np

def update_coefficients(initial_coeffs, constant_exponents, electron_density, grid):
    assert len(initial_coeffs) == len(constant_exponents)

    exponential = np.exp(-constant_exponents * np.power(grid, 2.))
    gaussian_density = np.dot(exponential, initial_coeffs)

    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))
    print(gaussian_density.shape, len(initial_coeffs))
    new_coefficients = np.empty(len(initial_coeffs))
    for i in range(0, len(initial_coeffs)):
        prefactor = initial_coeffs[i] * (constant_exponents[i] / np.pi)**(3/2)
        integrand = masked_electron_density * np.ravel(np.ma.asarray(np.exp(- constant_exponents[i] * np.power(grid, 2.))))\
                    / masked_gaussian_density
        new_coefficients[i] = prefactor * np.trapz(y=integrand, x=np.ravel(grid))
    return new_coefficients

if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 1000; NUMBER_OF_DIFFUSED_PTS = 1000
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)







