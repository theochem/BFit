from fitting.fit.least_sqs import *
from fitting.radial_grid.radial_grid import RadialGrid, HortonGrid
from fitting.density.gaussian_density.total_gaussian_dens import GaussianTotalBasisSet
from fitting.gbasis.gbasis import UGBSBasis
from fitting.fit.mbis_total_density import TotalMBIS
from fitting.density.slater_density.atomic_slater_density import Atomic_Density
from fitting.density.density_model import DensityModel
from fitting.fit.greedy_utils import GreedyMBIS, GreedyLeastSquares
import matplotlib.pyplot as plt
import horton
import warnings
import os
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='helvetica')


def get_next_possible_coeffs_and_exps(factor, coeffs, exps):
    size = exps.shape[0]
    all_choices_of_exponents = []
    all_choices_of_coeffs = []
    all_choices_of_parameters = []
    coeff_value = 100.
    for index, exp in np.ndenumerate(exps):
        if index[0] == 0:
            exponent_array = np.insert(exps, index, exp / factor)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        elif index[0] <= size:
            exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]]) / 2)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        #all_choices_of_exponents.append(exponent_array)
        #all_choices_of_coeffs.append(coefficient_array)
        all_choices_of_parameters.append(np.append(coefficient_array, exponent_array))
        if index[0] == size - 1:
            exponent_array = np.append(exps, np.array([exp * factor]))
            #all_choices_of_exponents.append(exponent_array)
            #all_choices_of_coeffs.append()
            all_choices_of_parameters.append(np.append(np.append(coeffs, np.array([coeff_value])), exponent_array))
    return all_choices_of_parameters #all_choices_of_coeffs, all_choices_of_exponents

def get_next_possible_coeffs_and_exps2(factor, coeffs, exps):
    size = exps.shape[0]
    all_choices_of_exponents = []
    all_choices_of_coeffs = []
    coeff_value = 100.
    for index, exp in np.ndenumerate(exps):
        if index[0] == 0:
            exponent_array = np.insert(exps, index, exp / factor)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        elif index[0] <= size:
            exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]]) / 2)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        all_choices_of_exponents.append(exponent_array)
        all_choices_of_coeffs.append(coefficient_array)
        if index[0] == size - 1:
            exponent_array = np.append(exps, np.array([exp * factor]))
            all_choices_of_exponents.append(exponent_array)
            all_choices_of_coeffs.append(np.append(coeffs,np.array([coeff_value])))
    return all_choices_of_coeffs, all_choices_of_exponents

def get_two_next_possible_coeffs_and_exps(factor, coeffs, exps):
    size = len(exps)
    all_choices_of_coeffs = []
    all_choices_of_exps = []
    coeff_value = 100.

    for index, exp in np.ndenumerate(exps):
        if index[0] == 0:
            exponent_array = np.insert(exps, index, exp / factor)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        elif index[0] <= size:
            exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]]) / 2)
            coefficient_array = np.insert(coeffs, index, coeff_value)

        coeff2, exp2 = get_next_possible_coeffs_and_exps2(factor, coefficient_array, exponent_array)
        all_choices_of_coeffs.extend(coeff2)
        all_choices_of_exps.extend(exp2)

        if index[0] == size - 1:
            exponent_array = np.append(exps, np.array([exp * factor]))
            coeff2, exp2 = get_next_possible_coeffs_and_exps2(factor, np.append(coeffs, np.array([coeff_value])),
                                                             exponent_array)
            all_choices_of_coeffs.extend(coeff2)
            all_choices_of_exps.extend(exp2)
    all_choices_params = []
    for i, c in enumerate(all_choices_of_coeffs):
        all_choices_params.append(np.append(c, all_choices_of_exps[i]))
    return all_choices_params

def pick_two_lose_one(factor, coeffs, exps):
    all_choices = []
    coeff_value = 100.
    two_choices = get_two_next_possible_coeffs_and_exps(factor, coeffs, exps)
    for i, p in enumerate(two_choices):
        coeff, exp = p[:len(p)//2], p[len(p)//2:]
        for j in range(0, len(p)//2):
            new_coeff = np.delete(coeff, j)
            new_exp = np.delete(exp, j)
            all_choices.append(np.append(new_coeff, new_exp))
    return all_choices

def plot_error(errors, element_name, title, figure_name):
    numb_of_errors = len(errors)
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    f, axarr = plt.subplots(2, 2, figsize=(12, 14), sharex=True)
    plt.suptitle(title, fontsize=17)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    for x in [0, 1]:
        for y in [0, 1]:
            axarr[x, y].spines["top"].set_visible(False)
            axarr[x, y].spines["bottom"].set_visible(False)
            axarr[x, y].spines["right"].set_visible(False)
            axarr[x, y].spines["left"].set_visible(False)
            axarr[x, y].get_xaxis().tick_bottom()
            axarr[x, y].get_yaxis().tick_left()
    # xrange(1, len(errors[2]) + 1, 2)
    #[1] + [x + x - 1  for x in range(2, len(errors[2]) + 1)]
    axarr[0, 0].plot(xrange(1, len(errors[2]) + 1, 1), errors[0], 'o-', color=tableau20[0])
    axarr[0, 0].set_title('Integrated Fitted Density Model')
    axarr[0, 0].set_ylabel(r'$\int \rho^o(r) 4 \pi r^2 dr$')
    axarr[0, 1].semilogy(xrange(1, len(errors[2]) + 1, 1), errors[1], 'o-', color=tableau20[0])
    axarr[0, 1].set_title('Absolute Difference In Models')
    axarr[0, 1].set_ylabel(r'$\int |\rho(r) - \rho^o(r)| dr$')
    axarr[1, 0].semilogy(xrange(1, len(errors[2]) + 1, 1), errors[2], 'o-', color=tableau20[0])
    axarr[1, 0].set_title("Absolute Difference Times Radius Squared")
    axarr[1, 0].set_xlabel("Iterations")
    axarr[1, 0].set_ylabel(r'$\int |\rho(r) - \rho^o(r)| r^2 dr$')
    axarr[1, 1].semilogy(xrange(1, len(errors[2]) + 1, 1), errors[3], 'o-', color=tableau20[0])
    axarr[1, 1].set_title('Kullback-Leiger Function Value')
    axarr[1, 1].set_ylabel(r'$\int \rho(r) \frac{\rho(r)}{\rho^o(r)} 4 \pi r^2 dr$')
    axarr[1, 1].set_xlabel("Iterations")
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    directory = os.path.dirname(__file__).rsplit('/', 2)[0] + "/fitting/results/" + element_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/" + figure_name + ".png")
    plt.close()


def plot_model_densities(true_dens, model_dens, grid_pts, title, element_name,
                         figure_name, additional_models_plots=None):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    plt.figure(figsize=(12, 14))
    ax = plt.subplot(111)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.semilogy(grid_pts, model_dens, '-', lw=3, label="Final Gaussian Fitted Electron Density",
                color=(214/255., 39/255., 40/255.))
    ax.semilogy(grid_pts, true_dens, '-', lw=3, label="Slater Electron Density",
                color=(31/255., 119/255., 180/255.))
    if additional_models_plots is not None:
        for i, model in enumerate(additional_models_plots):
            if i == 0:
                ax.semilogy(grid_pts, model_dens, '-', lw=1, label="Gaussian Fitted Electron Density",
                        color=tableau20[7])
            else:
                ax.semilogy(grid_pts, model_dens, '-', lw=1, color=tableau20[7])
    # plt.xlim(0, 25.0*0.5291772082999999)
    plt.xlim(0, 9)
    plt.ylim(ymin=1e-9)
    ax.set_axis_bgcolor('white')
    plt.xlabel('Distance from the nucleus [A]')
    plt.ylabel(r'$Log(\rho(r_{Bohr}^{-3}))$', fontsize=16)
    plt.title(title, fontweight='bold')
    plt.grid(color=tableau20[-2])
    plt.legend()
    directory = os.path.dirname(__file__).rsplit('/', 2)[0] + "/fitting/results/" + element_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/" + figure_name + ".png")
    #plt.show()
    plt.close()

def fit_radial_densities(element_name, atomic_number, grid=None, true_density=None,
                         density_model=None, method="SLSQP", options=None,
                         UGBS_type='S', ioutput=False, iplot=False):
    """
    Fits Radial Densities between different models.

    Parameters
    ----------
    element_name : str

    atomic_number : int
        atomic number of the atomic density.

    true_density : arr, optional
        Electron Density to be fitted from

    density_model : DensityModel, optional
        This is where you model and cost function is stored

    grid : arr or RadialGrid, optional
        grid evaulated
        default is horton.

    method : str or callable, optional
        Type of solver.  Should be one of
            - 'slsqp' :ref:`(see here) <fit.least_sqs.optimize_using_slsqp>`
            - 'l-bfgs'      :ref:`(see here) <fit.least_sqs.optimize_using_l_bfgs>`
            - 'nnls'          :ref:`(see here) <fit.least_sqs.optimize_using_nnls>`
            - 'greedy-ls-sqs'   :ref:
            - 'mbis'        :ref:`(see here) <>`
            - 'greedy-mbis'   :ref:`(see here) <>`

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.

    options : dict, optional
        - 'slsqp' - {bounds=(0, np.inf), initial_guess=custom(see *)}
        - 'l-bfgs' - {bounds=(0, np.inf), initial_guess=custom(see *)}
        - 'nnls' - {initial_guess=UGBS Exponents}
        - 'mbis' - {threshold_coeff, threshold_exp, initial_guess, iprint=False}
        - 'greedy-mbis' - {factor, max_number_of_functions, additional_funcs,
                           threshold_coeff, threshold_exp, splitting_func}
        - 'greedy-ls-sqs' - {factor, max_numb_of_funcs, additional_funcs,
                             splitting_func, threshold_coeff, threshold_exp}

        * initial guess is obtained by optimizing coefficients using NNLS using
        UGBS s-type exponents.

    UGBS_Type : str, optional
        default is 'S'
        denotes which type of UGBS exponents to get.

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
    if options is None:
        options = {}

    full_names = {"be":"Beryllium", "c":"Carbon", "he":"Helium", "li":"Lithium", 'b':"boron",
                  "n":"nitrogen", "o":"oxygen", "f":"fluoride", "ne":"neon"}
    element_name = element_name.lower()
    if grid is None:
        grid = HortonGrid(1.0e-30, 25, 1000)

    # Sets Grid array to become one of our grid objects
    if not isinstance(grid, (RadialGrid, HortonGrid)):
        warnings.warn("Integration is done by multiplying density by 4 pi radius squared", RuntimeWarning)
        grid = RadialGrid(grid)

    # Sets Default Density To Atomic Slater Density
    if true_density is None:
        file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/fitting/data/examples/' + element_name.lower()
        true_density = Atomic_Density(file_path, grid.radii).electron_density

    # Sets Default Density Model to Gaussian Density
    if density_model is None:
        file_path = os.path.dirname(__file__).rsplit('/', 2)[0] + '/fitting/data/examples/' + element_name.lower()
        density_model = GaussianTotalBasisSet(element_name, grid.radii, electron_density=true_density,
                                              file_path=file_path)

    # Exits If Custom Density Model is not inherited from density_model
    assert isinstance(density_model, DensityModel), "Custom Density Model should be inherited from " \
                                                    "DensityModel from density_model.py"

    # Gives Warning if you wanted a custom density model to mbis related procedures.
    if method in ["mbis", "greedy-mbis"] and density_model is not None:
        warnings.warn("Method %s does not use custom density models. Rather it uses default "
                      "gaussian density" % method, RuntimeWarning)

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
    if method in ['greedy-ls-sqs', 'greedy-mbis']:
        options.setdefault('factor', 2.)
        options.setdefault('max_numb_of_funcs', 30)
        options.setdefault('backward_elim_funcs', None)
        #options.setdefault('threshold_coeff', 1e-3)
        #options.setdefault('threshold_exps', 1e-4)
        #options.setdefault('splitting_func', get_next_possible_coeffs_and_exps)

    if method == "slsqp":
        params = optimize_using_slsqp(density_model, **options)
    elif method == "l-bfgs":
        params = optimize_using_l_bfgs(density_model, **options)
    elif method == "nnls":
        cofactor_matrix = density_model.create_cofactor_matrix(options['initial_guess'])
        params = optimize_using_nnls(cofactor_matrix)
    elif method == "mbis":
        mbis_obj = TotalMBIS(element_name, atomic_number, grid, true_density)
        params = mbis_obj.run(**options)
    elif method == "greedy-ls-sqs":
        pass
    elif method == "greedy-mbis":
        greedy_mbis = GreedyMBIS(element_name, atomic_number, grid, true_density, splitting_func=get_next_possible_coeffs_and_exps)

        if ioutput:
            params, params_it = greedy_mbis.run_greedy(ioutput=ioutput, **options)
            error = greedy_mbis.errors
            exit_info = greedy_mbis.exit_info
        else:
            params = greedy_mbis.run_greedy(ioutput=ioutput, **options)
        print("params", params)


    if iplot:
        # Change Grid To Angstrom
        grid.radii *= 0.5291772082999999
        model = greedy_mbis.mbis_obj.get_normalized_gaussian_density(params[:len(params)//2],
                                                                     params[len(params)//2:])
        plot_model_densities(true_density, model, grid.radii,
                             title="Electron Density Plot of " + full_names[element_name],
                             element_name=element_name,
                             figure_name="model_plot_using_" + method)
        models_it = []
        for p in params_it:
            c, e = p[:len(p)//2], p[len(p)//2:]
            models_it.append(greedy_mbis.mbis_obj.get_normalized_gaussian_density(c, e))
        plot_model_densities(true_density, model, grid.radii,
                             title="Electron Density Plot of " + full_names[element_name],
                             element_name=element_name,
                             figure_name="greedy_model_plot_using_" + method,
                             additional_models_plots=models_it)
        plot_error(error, element_name, "Different Error Measures On " + full_names[element_name],
                   figure_name="error_plot_using_" + method)

    if ioutput:
        dir = os.path.dirname(__file__).rsplit('/', 2)[0] + '/fitting/results/' + element_name
        file_object = open(dir + '/arguments_' + method + ".txt", "w+")
        file_object.write("Method Used " + method + "\n")
        file_object.write("Number Of Basis FUnctions" + str(len(params)//2) + "\n")
        file_object.write("Final Parameters: " + str(params) + "\n")
        file_object.write("Iteration Parameters: " + str(params_it) + "\n")
        file_object.write(str(options) + "\n")
        file_object.write("Exit Information: " + str(exit_info) + "\n")
        file_object.write("Redudandance Info: " + str(greedy_mbis.redudan_info_numb_basis_funcs))
        file_object.close()
        np.save(dir + "/parameters_" + method + ".npy", params)
        np.save(dir+"/parameters_" + method + "_iter.npy", params_it)

    return params

if __name__ == "__main__":
    for i, ele in enumerate(["he", "li", "be", "b", "c", "n", "o", "f", "ne"]):
        j = i + 2
        fit_radial_densities(ele, j, method='greedy-mbis', options={'max_numb_of_funcs': 2},
                             iplot=True, ioutput=True)
        raise ValueError

