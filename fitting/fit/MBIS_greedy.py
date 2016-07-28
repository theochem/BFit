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
    be_abs = GaussianSecondObjTrapz(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=1., lam2=0.)
    fitting_squared = Fitting(be_squared)
    fitting_abs = Fitting(be_abs)
    #params = fitting_squared.forward_greedy_algorithm(10000., 0.0001, be.electron_density)
    #print(params)
    fitting_obj = Fitting(be)
    print("True Electron Density ", be.integrated_total_electron_density)


    exps = be.UGBS_s_exponents
    print("Number of Exponents", len(exps))
    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))



    params = fitting_abs.optimize_using_slsqp(np.append(coeffs, exps), len(exps), additional_constraints=True)
    coeffs = params[0:len(exps)]
    exps = params[len(exps):]
    error = be.measure_error_by_integration_of_difference(be.electron_density, be.create_model(params, len(exps)))
    coeffs[coeffs == 0.] = 1e-6
    exps[exps == 0.] = 1e-6
    y = 0
    previous_error = 1e10
    keep_coeff = True
    keep_exps = True
    error_counter = 0

    while True:
        counter = 0

        for x in range(0, 50):
            coeffs = update_coefficients(coeffs, exps, be.electron_density, be.grid) ###

            exps = update_exponents(coeffs, exps, be.electron_density, be.grid)
            params = np.append(coeffs, exps)
            mod = be.create_model(params, len(exps))
            error = be.measure_error_by_integration_of_difference(be.electron_density, mod)
            print(counter, be.integrate_model_using_trapz(mod), error,
                            be.cost_function(np.append(coeffs, exps), len(exps)),be.measure_error_by_difference_of_integration(be.electron_density, mod)
                  ,y)
            previous_error = error
            counter += 1

        for x in range(0, 800):
            if keep_coeff:
                coeffs = update_coefficients(coeffs, exps, be.electron_density, be.grid) ###
            if keep_exps:
                exps = update_exponents(coeffs, exps, be.electron_density, be.grid)

            params = np.append(coeffs, exps)
            mod = be.create_model(params, len(exps))
            error = be.measure_error_by_integration_of_difference(be.electron_density, mod)
            print(counter, be.integrate_model_using_trapz(mod), error,
                            be.cost_function(np.append(coeffs, exps), len(exps)),be.measure_error_by_difference_of_integration(be.electron_density, mod)
                  ,y, keep_coeff, keep_exps)
            if error > previous_error:
                error_counter += 1
                if error_counter == 5:
                    if keep_coeff == True:
                        keep_coeff = False
                    elif keep_exps == True:
                        keep_exps = False
                        keep_coeff = True
                    error_counter = 0
            else:
                error_counter = 0
            previous_error = error
            counter += 1

        print(coeffs,exps)
        params = fitting_abs.optimize_using_slsqp(params, len(exps), additional_constraints=True)
        coeffs = params[0:len(exps)]
        exps = params[len(exps):]
        coeffs[coeffs == 0.] = 1e-6
        exps[exps == 0.] = 1e-6
        y += 1
        keep_coeff = True
        keep_exps = True