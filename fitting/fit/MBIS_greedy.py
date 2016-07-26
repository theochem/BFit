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
    be_squared = GaussianSecondObjSquared(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=1., lam2=0.)
    fitting_squared = Fitting(be_squared)
    #params = fitting_squared.forward_greedy_algorithm(100., 0.0001, be.electron_density)
    #print(params)
    fitting_obj = Fitting(be)

    coeffs = np.array([  4.66215508e-01  , 1.00000000e-6 ,  1.00000000e-6 ,  5.58743790e+01,
                           1.09927456e+01,   2.55667801e-02  , 3.78797496e-02 ,  7.31494431e+01,
                           2.02524030e-02  , 1.31802205e+01 ,  2.54901423e+01  , 6.30344728e+01,
                           5.16507491e+01 ,  1.38853868e-01 ,  5.31731590e+01  , 1.88839897e-04,
                           2.25881541e+01  , 1.85621159e-03 ,  1.26350382e-03 ,  6.98465108e-02,
                           9.83312716e-02  , 9.67229197e-02 ,  2.48424461e+01 ,  7.38026255e+00,
                           1.70683030e-01 ,  1.67290128e-01 ,  1.53982221e-01 ,  7.10964060e+00,
                           4.22493864e+00 ,  5.06618259e-02 ,  1.17920981e+01,   1.76819671e-02,
                           1.49243235e-19,   1.87018101e+01  ])





    exps = np.array([2.37124202e-01 ,  6.40113917e+00,
                           9.59254172e+00  , 1.28085647e+01 ,  2.85842540e+01  , 4.42998279e+01,
                           3.47058795e+01  , 3.02115299e+01  , 2.51464439e+01 ,  1.35496718e+01,
                           5.34768406e+00 ,  6.32198347e+01 ,  1.16142950e+02 ,  1.84890535e+02,
                           2.54203657e+02 ,  3.88014889e+02 ,  6.17290648e+02 ,  8.46589567e+02,
                           1.07586350e+03  , 1.19050120e+03 ,  1.30513891e+03 ,  1.76309827e+03,
                           2.22106588e+03 ,  2.22950520e+05 ,  1.67212910e+05 ,  1.11475301e+05,
                           5.57376895e+04 ,  2.78688852e+04,   1.39344832e+04 ,  1.04508829e+04,
                           6.96728323e+03 ,  8.03216207e-02  , 8.04560066e+00 ,  8.04581777e+02]  )

    y = 0
    while True:
        counter = 0
        for x in range(0, 300):
            exps = update_exponents(coeffs, exps, be.electron_density, be.grid)
            coeffs = update_coefficients(coeffs, exps, be.electron_density, be.grid) ###

            params = np.append(coeffs, exps)
            mod = be.create_model(params, len(exps))
            print(counter, be.integrate_model_using_trapz(mod), be.measure_error_by_integration_of_difference(be.electron_density, mod),
                            be.cost_function(np.append(coeffs, exps), len(exps)), y)
            counter += 1
        print(coeffs,exps)
        y += 1