import numpy as np
from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
from fitting.fit.MBIS import KL_objective_function
from scipy.optimize import broyden2

def get_exponents(electron_density, grid, atomic_number, coeff):
    return -(np.trapz(y=np.ravel(electron_density) * np.log(np.ravel(electron_density) / coeff), x=np.ravel(grid))) / atomic_number


if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 300; NUMBER_OF_DIFFUSED_PTS = 400
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    fitting_obj = Fitting(be)

    coeff = be.electron_density[0]
    exp = get_exponents(be.electron_density, be.grid, ATOMIC_NUMBER, coeff)
    print(coeff, exp)
    model = be.create_model(np.append(coeff, exp), 1)
    print(be.integrate_model_using_trapz(model))
    kl = KL_objective_function(coeff, exp, be.electron_density, be.grid)
    print(kl)
    def KL_objective_function2(parameters, true_density=be.electron_density, grid=be.grid):
        coefficients = parameters[:len(parameters)//2]
        exponents = parameters[len(parameters)//2:]
        exponential = np.exp(-exponents * np.power(grid, 2.))
        gaussian_density = np.dot(exponential, coefficients)

        masked_gaussian_density = np.ma.asarray(gaussian_density)
        masked_electron_density = np.ma.asarray(np.ravel(true_density))
        #masked_gaussian_density[masked_gaussian_density <= 1e-6] = 1e-12

        ratio = masked_electron_density / masked_gaussian_density
        return np.trapz(y=masked_electron_density * np.log(masked_electron_density) -
                        masked_electron_density * np.log(masked_gaussian_density), x=np.ravel(grid))
    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(be.UGBS_s_exponents))
    coeffs[coeffs == 0.] = 0.
    print(broyden2(KL_objective_function2, np.array([-5 for x in range(0,30)])))

