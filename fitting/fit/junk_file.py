import numpy as np
from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
from fitting.fit.MBIS import KL_objective_function
from scipy.optimize import broyden2
from fitting.fit.normalized_mbis import MBIS
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
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])[1:]
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    fitting_obj = Fitting(be)
    coeffs = np.array([  7.67163334e-15,   3.81490190e-03,   4.58541643e-18,   2.36675454e-01,
                       1.26174567e-01 ,  1.20608719e-01,   1.88961187e-09 ,  6.62059483e-02,
                       6.34315834e-01 ,  9.08376837e-01,   7.14355583e-01 ,  5.41385482e-01,
                       3.16551329e-01 ,  1.56727505e-01,   8.47536805e-02 ,  4.57210564e-02,
                       2.26040275e-02 ,  1.04208945e-02,   5.66069946e-03 ,  2.84130951e-03,
                       1.43757892e-03 ,  6.64960536e-04,   3.53818357e-04 ,  1.72414813e-04,
                       9.15607124e-05 ,  4.18651473e-05,   2.22983242e-05 ,  1.05309750e-05,
                       5.80365083e-06 ,  2.60922355e-06,   1.29509724e-06 ,  7.79661885e-07,
                       3.43721155e-07 ,  8.39357998e-08,   1.82224296e-07 ,  1.42750438e-17,
                       3.85412742e-18 ,  4.34530002e-08] )
    exps = np.array([  7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03 ,7.82852278e+03,
                      7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 , 7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.86683468e+02,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03])
    be.electron_density /= 4 * np.pi
    weights = 1 / (4 * np.pi * row_grid_points)
    mbis_obj = MBIS(be, weights, ATOMIC_NUMBER)
    print(mbis_obj.lamda_multplier)
    model = mbis_obj.get_normalized_gaussian_density(coeffs, exps)
    masked_elec = np.ma.asarray(np.ravel(be.electron_density.copy()))
    masked_model = np.ma.asarray(model)
    masked_model[masked_model == 0.] = 1e-20
    ratio = masked_elec / masked_model
    print(np.trapz(y=np.ravel(be.electron_density), x=row_grid_points) / 4)
    print((exps[-1] / np.pi)**(3/2) * np.trapz(y=(ratio) * np.exp(-exps[-1] * np.power(row_grid_points, 2.))
                                                  ,   x=row_grid_points))