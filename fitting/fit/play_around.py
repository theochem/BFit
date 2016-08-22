import numpy as np

from mbis_abc import TotalMBIS
from fitting.density.radial_grid import Radial_Grid
from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet
import os

if __name__ == "__main__":
    ATOMIC_NUMBER = 3
    ELEMENT = "li"
    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    file_path = current_directory + "data\examples\\" + ELEMENT #+ ".slater"

    class Grid2():
        def __init__(self, number_of_points):
            self.radii =  np.arange(0.000001, 25, 1/number_of_points)
            self.column_radii = np.reshape(self.radii, (len(self.radii), 1))
        def integrate(self, arr):
            return np.trapz(y=4. * np.pi * np.power(self.radii, 2.) * arr , x=self.radii)

    g = Grid2(10000)
    atom = GaussianTotalBasisSet(ELEMENT, g.column_radii, file_path)
    atom.electron_density /= 4. * np.pi
    print(g.integrate(np.ravel(atom.electron_density)))

    mbis = TotalMBIS(atom.electron_density, g, np.ones(len(g.radii)), ATOMIC_NUMBER, ELEMENT)
    coeffs = np.array([  6.86514754e-09,   1.94652163e-14,   4.22268838e-14,   1.35607284e-08,
           1.96906701e-07 ,  6.40276596e-08 ,  4.27626251e-07 ,  1.87502382e-06,
           1.36279592e-05 ,  2.79344860e-05 ,  5.36325928e-04 ,  1.09872578e-03,
           4.21852635e-03 ,  1.54545672e-02 ,  6.84464687e-02 ,  1.05494620e-01,
           1.94599971e-01 ,  7.62369844e-01 ,  7.74957086e-01 ,  2.72617418e-03,
           3.02081751e-11 ,  3.09561635e-10 ,  7.76627414e-01 ,  2.86164365e-01,
           7.26176438e-03])

    exps = np.array([  2.25007972e+05   ,2.25007972e+05   ,2.25007972e+05   ,2.03424205e+04,
           2.03424205e+04  , 2.03424205e+04  , 2.03424205e+04  , 2.44766583e+03,
           2.44766583e+03  , 2.44766583e+03  , 3.69104588e+02  , 3.69104588e+02,
           8.25234465e+01  , 8.25229397e+01  , 3.35774347e+01  , 1.43920905e+01,
           1.39984426e+01  , 5.67032841e+00  , 2.30228007e+00  , 2.30228007e+00,
           2.30228007e+00  , 1.14570518e-01  , 1.14570518e-01  , 5.82543712e-02,
           3.09679556e-02])
    print(mbis.get_descriptors_of_model(mbis.get_normalized_gaussian_density(coeffs, exps)))