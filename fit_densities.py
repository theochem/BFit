r"""

"""

import os
import warnings
from fitting.least_squares.least_sqs import *
from fitting.radial_grid.radial_grid import ClenshawGrid, HortonGrid
from fitting.least_squares.gaussian_density.gaussian_dens import \
    GaussianBasisSet
from fitting.gbasis.gbasis import UGBSBasis
from fitting.kl_divergence.gaussian_kl import GaussianKullbackLeibler
from fitting.least_squares.slater_density.atomic_slater_density import \
    Atomic_Density
from fitting.least_squares.density_model import DensityModel
from fitting.greedy.greedy_kl import GreedyKL
from fitting.utils.plotting_utils import plot_model_densities, plot_error


__all__ = ["fit_radial_densities"]


def get_hydrogen_electron_density(grid, bohr_radius=1):
    return (1. / np.pi * (bohr_radius ** 3.)) * np.exp(-2. * grid / bohr_radius)


def fit_radial_densities(element_name, atomic_number, grid=None, true_density=None,
                         density_model=None, method="SLSQP", options=None,
                         UGBS_type='S', ioutput=False, iplot=False):
    """
    Fits Radial Densities between different models.

    Parameters
    ----------
    element_name : str
                  The element that is being fitted to.
    atomic_number : int
        atomic number of the element.

    true_density : arr, optional
        Electron Density to be fitted from

    density_model : DensityModel, optional
        This is where your model and cost function is stored for least squares.

    grid : arr or ClenshawGrid or RadialGrid or HortonGrid, optional
        _grid evaulated
        default is horton.

    method : str or callable, optional
        Type of solver.  Should be one of
            - 'slsqp' :ref:`(see here) <kl_divergence.least_sqs.optimize_using_slsqp>`
            - 'l-bfgs'      :ref:`(see here) <kl_divergence.least_sqs.optimize_using_l_bfgs>`
            - 'nnls'          :ref:`(see here) <kl_divergence.least_sqs.optimize_using_nnls>`
            - 'greedy-ls-sqs'   :ref:'(see here) <greedy.greedy_lq.GreedyLeastSquares>'
            - 'kl_divergence'        :ref:`(see here) <kl_divergence.kull_leib_fitting>`
            - 'greedy-kl_divergence'   :ref:`(see here) <greedy.greedy_kl.GreedyKL>`

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.

    options : dict, optional
        - 'slsqp' - {bounds=(0, np.inf), initial_guess=custom(see *)}
        - 'l-bfgs' - {bounds=(0, np.inf), initial_guess=custom(see *)}
        - 'nnls' - {initial_guess=UGBS Exponents}
        - 'kl_divergence' - {threshold_coeff, threshold_exp, initial_guess, iprint=False}
        - 'greedy-kl_divergence' - {factor, max_number_of_functions, additional_funcs,
                           threshold_coeff, threshold_exp, splitting_func}
        - 'greedy-ls-sqs' - {factor, max_numb_of_funcs, additional_funcs,
                             splitting_func, threshold_coeff, threshold_exp}

        * initial guess is obtained by optimizing coefficients using NNLS using
        UGBS s-type exponents.

    UGBS_Type : str, optional
        default is 'S'
        denotes which type of UGBS exponents to get.

    iplots : boolean, optional

    ioutput : boolean, optional

    Returns
    -------

    Notes
    -----

    References
    ----------

    Examples
    --------

    """
    if options is None:
        options = {}

    full_names = {"be": "Beryllium", "c": "Carbon", "he": "Helium", "li": "Lithium",
                  'b': "boron", "n": "nitrogen", "o": "oxygen", "f": "fluoride",
                  "ne": "neon"}

    element_name = element_name.lower()
    if grid is None:
        #_grid = HortonGrid(1.0e-30, 25, 1000)
        pass

    # Sets Grid array to become one of our _grid objects
    if not isinstance(grid, (ClenshawGrid, HortonGrid)):
        warnings.warn("Integration is done by multiplying least_squares by 4 pi radius squared", RuntimeWarning)
        grid = ClenshawGrid(grid, None, None)

    # Sets Default Density To Atomic Slater Density
    if true_density is None:
        file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/fitting/data/examples/' + element_name.lower()
        true_density = Atomic_Density(file_path, grid.radii).electron_density

    # Sets Default Density Model to Gaussian Density
    if density_model is None:
        file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/fitting/data/examples/' + element_name.lower()
        density_model = GaussianBasisSet(element_name, grid.radii, elec_dens=true_density,
                                         file_path=file_path)

    # Exits If Custom Density Model is not inherited from density_model
    assert isinstance(density_model, DensityModel), "Custom Density Model should be inherited from " \
                                                    "DensityModel from density_model.py"

    # Gives Warning if you wanted a custom density model to kl_divergence related procedures.
    if method in ["kl_divergence", "greedy-kl_divergence"] and density_model is not None:
        warnings.warn("Method %s does not use custom least_squares models. Rather it uses default "
                      "gaussian least_squares" % method, RuntimeWarning)

    # Sets Initial Guess For Exponents to S-type UGBS and then uses NNLS to get coefficients as initial guess
    if method in ['slsqp', 'l-bfgs']:
        options.setdefault('bounds', (0, np.inf))
        ugbs_exps = UGBSBasis(element_name).exponents(UGBS_type)
        cofactor_matrix = density_model.create_cofactor_matrix(ugbs_exps)
        params = optimize_using_nnls(cofactor_matrix)
        options.setdefault('initial_guess',
                           np.append(params, UGBSBasis(element_name).exponents(UGBS_type)))

    # Sets Exponents needed for NNLS to S-type UGBS
    if method == "nnls":
        ugbs_exps = UGBSBasis(element_name).exponents(UGBS_type)
        options.setdefault('initial_guess', ugbs_exps)

    # Set Default Arguments For Greedy
    if method in ['greedy-ls-sqs', 'greedy-kl_divergence']:
        options.setdefault('factor', 2.)
        options.setdefault('max_numb_of_funcs', 30)
        options.setdefault('backward_elim_funcs', None)
        #options.setdefault('splitting_func', get_next_choices)

    if method == 'kl_divergence':
        options.setdefault('threshold_coeff', 1e-3)
        options.setdefault('threshold_exps', 1e-4)
        options.setdefault('coeff_arr', options['coeff_arr'])
        options.setdefault('exp_arr', options['exp_arr'])

    if method == "slsqp":
        params = optimize_using_slsqp(density_model, **options)
    elif method == "l-bfgs":
        params = optimize_using_l_bfgs(density_model, **options)
    elif method == "nnls":
        cofactor_matrix = density_model.create_cofactor_matrix(options['initial_guess'])
        params = optimize_using_nnls(cofactor_matrix)
    elif method == "kl_divergence":
        mbis_obj = GaussianKullbackLeibler(element_name, atomic_number, grid, true_density)
        params = mbis_obj.__call__(**options)
    elif method == "greedy-ls-sqs":
        pass
    elif method == "greedy-kl_divergence":
        greedy_mbis = GreedyKL(element_name, atomic_number, grid, true_density,
                               splitting_func=pick_two_lose_one)
        if ioutput:
            params, params_it = greedy_mbis.__call__(ioutput=ioutput, **options)
            error = greedy_mbis.errors
            exit_info = greedy_mbis.exit_info
        else:
            params = greedy_mbis.__call__(ioutput=ioutput, **options)

    if iplot:
        # Change Grid To Angstrom
        grid.radii *= 0.5291772082999999
        model = greedy_mbis.mbis_obj.get_model(params[:len(params) // 2],
                                                                     params[len(params)//2:])
        plot_model_densities(greedy_mbis.mbis_obj.electron_density, model, grid.radii,
                             title="Electron Density Plot of " + full_names[element_name],
                             element_name=element_name,
                             figure_name="model_plot_using_" + method)
        models_it = []
        for p in params_it:
            c, e = p[:len(p)//2], p[len(p)//2:]
            models_it.append(greedy_mbis.mbis_obj.get_model(c, e))
        plot_model_densities(greedy_mbis.mbis_obj.electron_density, model, grid.radii,
                             title="Electron Density Plot of " + full_names[element_name],
                             element_name=element_name,
                             figure_name="greedy_model_plot_using_" + method,
                             additional_models_plots=models_it)
        plot_error(error, element_name, "Different Error Measures On " + full_names[element_name],
                   figure_name="error_plot_using_" + method)

    if ioutput:
        # dir = os.path.dirname(__file__).rsplit('/', 2)[0] + '/fitting/results_redudancies_two_lose_one/' + element_name
        dir = "/work/tehrana/fitting/fitting/results_redudancies_two_lose_one/" + element_name
        file_object = open(dir + '/arguments_' + method + ".txt", "w+")
        file_object.write("Method Used " + method + "\n")
        file_object.write("Number Of Basis FUnctions: " + str(len(params)//2) + "\n")
        file_object.write("Final Parameters: " + str(params) + "\n")
        file_object.write("Iteration Parameters: " + str(params_it) + "\n")
        file_object.write(str(options) + "\n")
        file_object.write("Exit Information: " + str(exit_info) + "\n")
        file_object.write("Redudandance Info: " + str(greedy_mbis.redudan_info_numb_basis_funcs))
        file_object.close()
        np.save(dir + "/parameters_" + method + ".npy", params)
        np.save(dir+"/parameters_" + method + "_iter.npy", params_it)

    return params

