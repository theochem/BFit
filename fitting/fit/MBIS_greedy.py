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
    NUMBER_OF_CORE_POINTS = 600; NUMBER_OF_DIFFUSED_PTS = 700
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

    coeffs = np.array([  1.97825026e-02,   1.28609902e-02,   4.06924751e-03,   4.48384485e-01,
                       1.26804269e-06  , 1.19828139e-06,   7.40132339e-07,   2.55954617e+00,
                       2.20278554e+01  , 5.57824019e+01,   6.88742751e+01,   6.57275582e+01,
                       6.15874797e+01  , 4.26452017e+01,   4.20482136e+01,   3.01016562e+01,
                       1.29608829e+01  , 1.13144165e+01,   8.23558374e+00,   5.66614217e+00,
                       4.46949316e+00  , 2.93847557e+00,   2.08078318e+00,   1.41938220e+00,
                       1.13476132e+00  , 7.62188507e-01,   5.19308824e-01,   2.86977874e-01,
                       1.27739174e-01  , 9.16601750e-02,   4.45832174e-02,   3.11385533e-02,
                       8.39674190e-36  , 2.09109607e-35,   2.55993510e-36,   5.96819917e-08,
                       1.50981403e-05  , 6.84861794e+01])
    exps = np.array([  1.04214184e-01,   1.04214184e-01,   1.04214184e-01,   2.45057983e-01,
                       2.45057983e-01,   2.45057983e-01,   5.15308902e+00,   5.15308930e+00,
                       5.33218778e+00,   1.20047435e+01,   2.39594449e+01,   4.58245084e+01,
                       1.12322988e+02,   1.16500262e+02,   5.59694399e+02,   5.59780361e+02,
                       2.46875764e+03,   3.84457850e+03,   4.84926676e+03,   1.60444870e+04,
                       2.79808013e+04,   4.57364150e+04,   9.89737931e+04,   2.26496533e+05,
                       5.20957654e+05,   6.79032136e+05,   1.21755805e+06,   1.43699888e+06,
                       8.30191007e+06,   8.30191016e+06 ,  8.30191019e+06,   8.30191022e+06,
                       2.30269403e+08,   2.30269403e+08,   2.30269403e+08,   2.83585691e+10,
                       2.83585691e+10,   2.83585691e+10])

    exps = np.array([  2.50000000e-02,   5.00000000e-02,   1.00000000e-01,   2.00000000e-01,
                       4.00000000e-01,   8.00000000e-01,   1.60000000e+00,   3.20000000e+00,
                       6.40000000e+00,   1.28000000e+01,   2.56000000e+01,   5.12000000e+01,
                       1.02400000e+02,   2.04800000e+02,   4.09600000e+02,   8.19200000e+02,
                       1.63840000e+03,   3.27680000e+03,   6.55360000e+03,   1.31072000e+04,
                       2.62144000e+04,   5.24288000e+04,   1.04857600e+05,   2.09715200e+05,
                       4.19430400e+05,   8.38860800e+05,   1.67772160e+06,   3.35544320e+06,
                       6.71088640e+06,   1.34217728e+07,   2.68435456e+07,   5.36870912e+07,
                       1.07374182e+08,   2.14748365e+08,   4.29496730e+08,   8.58993459e+08,
                       1.71798692e+09,   3.43597384e+09])
    # 2.5 e12 2.5e-3 1.9

    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    params = np.append(coeffs, exps)
    error = be.measure_error_by_integration_of_difference(be.electron_density, be.create_model(params, len(exps)))
    print(error, be.integrate_model_using_trapz(be.create_model(params, len(exps))),
          be.measure_error_by_difference_of_integration(be.electron_density, be.create_model(params, len(exps))))
    #params = fitting_abs.optimize_using_slsqp(np.append(coeffs, exps), len(exps), additional_constraints=True)
    #coeffs = params[0:len(exps)]
    #exps = params[len(exps):]
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
            temp_coeffs = coeffs.copy()
            #coeffs = update_coefficients(coeffs, exps, be.electron_density, be.grid) ###

            exps = update_exponents(temp_coeffs, exps, be.electron_density, be.grid)
            params = np.append(coeffs, exps)
            mod = be.create_model(params, len(exps))
            error = be.measure_error_by_integration_of_difference(be.electron_density, mod)
            print(counter, be.integrate_model_using_trapz(mod), error,
                            be.cost_function(np.append(coeffs, exps), len(exps)),be.measure_error_by_difference_of_integration(be.electron_density, mod)
                  ,y)
            previous_error = error
            counter += 1

        for x in range(0, 800):
            temp_coeffs = coeffs.copy()

            if keep_coeff:
                coeffs = update_coefficients(coeffs, exps, be.electron_density, be.grid) ###
            if keep_exps:
                exps = update_exponents(temp_coeffs, exps, be.electron_density, be.grid)

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
                        keep_exps = True
                    elif keep_exps == True:
                        keep_exps = False
                        keep_coeff = True
                    error_counter = 0
            else:
                error_counter = 0
            previous_error = error
            counter += 1

        print(coeffs,exps)
        #params = fitting_abs.optimize_using_slsqp(params, len(exps), additional_constraints=True)
        #coeffs = params[0:len(exps)]
        #exps = params[len(exps):]
        coeffs[coeffs == 0.] = 1e-6
        exps[exps == 0.] = 1e-6
        y += 1
        keep_coeff = True
        keep_exps = True