from __future__ import division
from mbis_abc import MBIS_ABC
import numpy as np



if __name__ == "__main__":
    TOMIC_NUMBER = 9
    ELEMENT_NAME = "f"
    USE_HORTON = False
    USE_FILLED_VALUES_TO_ZERO = True
    THRESHOLD_COEFF = 1e-8
    THRESHOLD_EXPS = 40
    import os

    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    file_path = current_directory + "data/examples//" + ELEMENT_NAME
    if USE_HORTON:
        import horton

        rtf = horton.ExpRTransform(1.0e-30, 25, 1000)
        radial_grid_2 = horton.RadialGrid(rtf)
        from fitting.density.radial_grid import Horton_Grid

        radial_grid = Horton_Grid(1e-80, 25, 1000, filled=USE_FILLED_VALUES_TO_ZERO)
    else:
        NUMB_OF_CORE_POINTS = 400;
        NUMB_OF_DIFFUSE_POINTS = 500
        from fitting.density.radial_grid import Radial_Grid
        from fitting.density.atomic_slater_density import Atomic_Density

        radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMB_OF_CORE_POINTS, NUMB_OF_DIFFUSE_POINTS, [50, 75, 100],
                                  filled=USE_FILLED_VALUES_TO_ZERO)

    from fitting.density import Atomic_Density

    atomic_density = Atomic_Density(file_path, radial_grid.radii)
    from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet

    from fitting.fit.model import Fitting

    atomic_gaussian = GaussianTotalBasisSet(ELEMENT_NAME, np.reshape(radial_grid.radii,
                                                                     (len(radial_grid.radii), 1)), file_path)
    weights = None  # (4. * np.pi * radial_grid.radii**1.)#1. / (1 + (4. * np.pi * radial_grid.radii ** 2.))#1. / (4. * np.pi * radial_grid.radii**0.5) #np.exp(-0.01 * radial_grid.radii**2.)

    fitting_obj = Fitting(atomic_gaussian)