from fitting.fit.least_sqs import *
from fitting.radial_grid.radial_grid import RadialGrid, HortonGrid
from fitting.density.gaussian_density.total_gaussian_dens import GaussianTotalBasisSet
from fitting.gbasis.gbasis import UGBSBasis
import os


def fit_radial_densities(grid, element_name, atomic_number, true_density=None,
                         density_model=None, method="SLSQP", options=None,
                         ioutput=False, iplot=False):
    """
    Fits Radial Densities between different models.

    Parameters
    ----------
    grid : arr or RadialGrid
        grid evaulated

    element_name : str

    atomic_number : int
        atomic number of the atomic density.

    true_density : arr, optional
        Electron Density to be fitted from

    density_model : DensityModel, optional
        This is where you model and cost function is stored

    method : str or callable, optional
        Type of solver.  Should be one of
            - 'SLSQP' :ref:`(see here) <fit.least_sqs.optimize_using_slsqp>`
            - 'L-BFGS-B'      :ref:`(see here) <fit.least_sqs.optimize_using_l_bfgs>`
            - 'NNLS'          :ref:`(see here) <fit.least_sqs.optimize_using_nnls>`
            - 'greedy_ls_sqs'   :ref:
            - 'MBIS'        :ref:`(see here) <>`
            - 'greedy-mbis'   :ref:`(see here) <>`

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.

    options : dict, optional
        - 'SLSQP' - {bounds=(0, np.inf), initial_guess=UGBS Exponents, UGBS_Type='S'}
        - 'L-BFGS-B' - {bounds=(0, np.inf), initial_guess=UGBS Exponents, UGBS_Type='S'}
        - 'NNLS' - {initial_guess=UGBS Exponents, UGBS_Type='S'}


    plots : boolean, optional

    output : boolean, optional

    Returns
    -------

    Notes
    -----

    References
    ----------

    Examples
    --------

    """
    element_name = element_name.lower()
    if not isinstance(grid, RadialGrid) and not isinstance(grid, HortonGrid):
        grid = RadialGrid(grid)

    if density_model is None:
        file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/examples/' + element_name.lower() + '.slater'
        density_model = GaussianTotalBasisSet(element_name, grid.radii, electron_density=true_density,
                                              file_path=file_path)
    if method == 'SLSQP' or method == 'L-BFGS':
        if options['initial_guess'] is not None:
            if options['UGBS_Type'] is None:
                options['UGBS_Type'] = 'S'
            cofactor_matrix = density_model.create_cofactor_matrix(options['initial_guess'])
            params = optimize_using_nnls(cofactor_matrix)
            options['initial_guess'] = np.append(params, UGBSBasis(element_name).exponents(options['UGBS_Type']))

    if method == "NNLS":
        if options['initial_guess'] is not None:
            if options['UGBS_Type'] is None:
                options['UGBS_Type'] = 'S'
        cofactor_matrix = density_model.create_cofactor_matrix(options['initial_guess'])
        options['initial_guess'] = optimize_using_nnls(cofactor_matrix)

    if method == "SLSQP":
        params = optimize_using_slsqp(density_model, inital_guess=options['initial_guess'], bounds=options['bounds'])
    elif method == "L-BFGS":
        params = optimize_using_l_bfgs(density_model, initial_guess=options['initial_guess'], bounds=options['bounds'])
    elif method == "NNLS":
        cofactor_matrix = density_model.create_cofactor_matrix(options['initial_guess'])
        params = optimize_using_nnls(cofactor_matrix)
    elif method == "MBIS":
        pass
    elif method == "greedy_ls_sqs":
        pass
    elif method == "greedy_mbis":
        pass

    if iplot:
        pass
    if ioutput:
        pass

    return params

