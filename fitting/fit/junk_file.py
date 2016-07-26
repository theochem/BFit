import numpy as np
from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *

best_coeff = np.array([  4.86584684e+000,   1.94623947e-007,   1.57542509e-006,   5.93555077e+000,
                   3.76245327e+000,   4.33964378e-001 ,  1.53773578e+001 ,  1.06245413e+001,
                   5.82104583e+000,   2.78854284e+001 ,  4.59539708e+001 ,  1.47019354e+000,
                   8.92252051e+001,   4.08913241e+001 ,  8.46189876e+001 ,  5.23633144e+001,
                   4.28219003e+001,   1.13816047e+001 ,  3.57719574e-076 ,  4.67821388e-189,
                   4.13830225e-149,   3.30424163e-001 ,  1.61434546e-001 ,  6.88738582e-004,
                   4.06659491e-004] )
best_exps = np.array([  4.14995051e+05,   2.06268686e+05,   1.04657882e+05,   5.36434000e+04,
                   2.75646792e+04,   1.40825166e+04,   7.14971902e+03,   3.66499964e+03,
                   1.87502376e+03,   9.43696747e+02,   4.91815608e+02,   2.49474497e+02,
                   1.26815526e+02,   6.54521849e+01,   3.28776275e+01,   1.71987192e+01,
                   8.54415057e+00,   4.47415781e+00,   3.43881642e+00,   2.68655133e+00,
                   5.45126410e-01,   2.89698612e-01,   1.58424110e-01,   4.86452567e-02,
                   4.86452306e-02])

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

    best_model = be.create_model(np.append(best_coeff, best_exps), len(best_exps))
    print("Model Integration - ", be.integrate_model_using_trapz(best_model), "True Density - ",
                                    be.integrated_total_electron_density, "Integration Difference - ",
                                    np.abs(be.integrated_total_electron_density - be.integrate_model_using_trapz(\
                                        best_model)),
                                    "Error measure - ", be.measure_error_by_integration_of_difference(be.electron_density,
                                                                                                      best_model))
    print("maximum exponent - ", np.max(best_exps))
    print("minimum exponent - ", np.min(best_exps))
    print("maximum coeff - ", np.max(best_coeff))
    print("minimum coeff - ", np.min(best_coeff))
    print("Number of Core Points For Grid - ", 300)
    print("Number of Diffused Points For Grid - ", 400)
    print("Extra Points added - ", [50, 75, 100])





