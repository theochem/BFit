import numpy as np

from fitting.least_squares.atomic_density.atomic_slater_density import Atomic_Density

from fitting.kl_divergence.gaussian_kl import GaussianKullbackLeibler
from fitting.radial_grid.radial_grid import ClenshawGrid


def get_grid_obj(atomic_number, numb_of_core_points, numb_of_diff_points, extra_list=[50, 75, 100]):
    radial_grid = ClenshawGrid(atomic_number, numb_of_core_points, numb_of_diff_points, extra_list,
                               filled=True)
    return radial_grid

def get_electron_density(element_name, grid):
    import os
    current_directory = os.path.dirname(os.path.abspath(__file__))[:-8]
    file_path = current_directory + "data/examples//" + element_name

    return Atomic_Density(file_path, grid)

def get_mbis_object(weights, atomic_number, element_name):
    radial_grid = get_grid_obj(atomic_number, 300, 400)
    atomic_dens = get_electron_density(element_name, radial_grid.radii)
    mbis_abc = GaussianKullbackLeibler(element_name, atomic_number, radial_grid, atomic_dens.atomic_density(), weights)

    return mbis_abc

def test_lagrange_multipliers_with_no_weights():
    no_weight = None
    # Beryllium
    mbis = get_mbis_object(no_weight, 4, "be")
    assert np.abs(mbis._lagrange_multiplier - 1.) < 1e-5

    #Carbon
    mbis = get_mbis_object(no_weight, 6, 'c')
    assert np.abs(mbis._lagrange_multiplier - 1.) < 1e-5

    #Copper
    mbis = get_mbis_object(no_weight, 29, 'cu')
    assert np.abs(mbis._lagrange_multiplier - 1.) < 1e-5

    #Silver
    mbis = get_mbis_object(no_weight,47, 'ag')
    assert np.abs(mbis._lagrange_multiplier - 1.) < 1e-5

def test_lagrange_multiplier_with_constant_weight():
    # Berllium
    atomic_number = 4
    grid_obj = get_grid_obj(atomic_number, 400, 300)
    weight = 2. * np.ones(len(grid_obj.radii))
    mbis = get_mbis_object(weight, atomic_number, "be")
    assert np.abs(mbis._lagrange_multiplier - 2.) < 1e-5

    # Carbon
    atomic_number = 6
    grid_obj = get_grid_obj(atomic_number, 400, 300)
    weight = 4. * np.ones(len(grid_obj.radii))
    mbis = get_mbis_object(weight, atomic_number, "c")
    assert np.abs(mbis._lagrange_multiplier - 4.) < 1e-3

    # Copper
    atomic_number = 29
    grid_obj = get_grid_obj(atomic_number, 400, 300)
    weight = 4. * np.pi  * np.ones(len(grid_obj.radii))
    mbis = get_mbis_object(weight, atomic_number, "cu")
    assert np.abs(mbis._lagrange_multiplier - 4. * np.pi) < 1e-3

def test_lagrange_multiplier_with_four_pi_weight():
    # Beryllium
    atomic_number = 4
    grid_obj = get_grid_obj(atomic_number, 300, 400)

    weight = 1 / (4 * np.pi * np.power(grid_obj.radii, 2.))
    mbis = get_mbis_object(weight, atomic_number, "be")
    assert np.abs(mbis._lagrange_multiplier - np.trapz(y=mbis.ma_elect_dens, x=np.ma.asarray(grid_obj.radii)) / atomic_number) < 1e-2

    # Carbon
    atomic_number = 6
    grid_obj = get_grid_obj(atomic_number, 300, 400)

    weight = 1 / (4 * np.pi * np.power(grid_obj.radii, 2.))
    mbis = get_mbis_object(weight, atomic_number, "c")

    assert np.abs(mbis._lagrange_multiplier - np.trapz(y=mbis.ma_elect_dens, x=np.ma.asarray(grid_obj.radii)) / atomic_number) < 1e-2


if __name__ == "__main__":
    print ("TEST")
    test_lagrange_multipliers_with_no_weights()
    test_lagrange_multiplier_with_constant_weight()
    test_lagrange_multiplier_with_four_pi_weight()
