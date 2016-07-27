from fitting.fit.GaussianBasisSet import *
from fitting.fit.multi_objective import *
import numpy as np
from scipy.integrate import simps, trapz
from fitting.fit.MBIS import *



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
    be_squared = GaussianSecondObjSquared(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=0.1, lam2=0.)
    be_abs = GaussianSecondObjTrapz(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=0.1, lam2=0.)
    fitting_squared = Fitting(be_squared)
    fitting_abs = Fitting(be_abs)
    #params = fitting_squared.forward_greedy_algorithm(10000., 0.0001, be.electron_density)
    #print(params)
    fitting_obj = Fitting(be)

    exps = be.generation_of_UGBS_exponents(1.1, be.UGBS_s_exponents)
    print("Number of Exponents", len(exps))
    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))

    #params = fitting_abs.optimize_using_slsqp(np.append(coeffs, exps), len(exps), additional_constraints=True)
    #coeffs = params[0:len(exps)]
    #exps = params[len(exps):]
    coeffs[coeffs == 0.] = 1e-6
    y = 0
    while True:
        counter = 0
        for x in range(0, 800):
            exps = update_exponents(coeffs, exps, be.electron_density, be.grid)
            coeffs = update_coefficients(coeffs, exps, be.electron_density, be.grid) ###

            params = np.append(coeffs, exps)
            mod = be.create_model(params, len(exps))
            print(counter, be.integrate_model_using_trapz(mod), be.measure_error_by_integration_of_difference(be.electron_density, mod),
                            be.cost_function(np.append(coeffs, exps), len(exps)), y)
            counter += 1
        print(coeffs,exps)
        #params = fitting_abs.optimize_using_slsqp(params, len(exps), additional_constraints=True)
        #coeffs = params[0:len(exps)]
        #exps = params[len(exps):]
        #coeffs[coeffs == 0.] = 1e-6
        #exps[exps == 0.] = 1e-6
        y += 1