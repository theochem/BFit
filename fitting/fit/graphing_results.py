import matplotlib.pyplot as plt


def plot_density_sections(dens, prodens, points, atom_name, title='None'):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib import rcParams
    # choose fonts

    plt.rc('text', usetex=True)
    plt.rc('font', family='Helvetica')
    # plotting intervals
    sector = 1.e-5
    conditions = [points <= sector,
                  np.logical_and(points > sector, points <= 100 * sector),
                  np.logical_and(points > 100 * sector, points <= 1000 * sector),
                  np.logical_and(points > 1000 * sector, points <= 1.e5 * sector),
                  np.logical_and(points > 1.e5 * sector, points <= 2.e5 * sector),
                  np.logical_and(points > 2.e5 * sector, points <= 5.0),
                  np.logical_and(points > 5.0, points <= 10.0)]

    # plot within each interval
    for i, cond in enumerate(conditions):
        # setup figure
        fig, axes = plt.subplots(2, 1)

        # plot true & model density
        ax1 = axes[0]
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.plot(points[cond], dens[cond], 'ro', linestyle='-', label=r'True')
        ax1.plot(points[cond], prodens[cond], 'bo', linestyle='--', label=r'Approx')
        ax1.legend(loc=0, frameon=False)
        xmin, xmax = np.min(points[cond]), np.max(points[cond])
        ax1.set_xticks(ticks=np.linspace(xmin, xmax, 5))
        ymin, ymax = np.min(dens[cond]), np.max(dens[cond])
        ax1.set_yticks(ticks=np.linspace(ymin, ymax, 5))
        ax1.set_ylabel(r'Density')
        if np.any(points[cond] < 1.0):
            ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        # Hide the right and top spines
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.grid(True, zorder=0, color='0.60')

        # plot difference of true & model density
        ax2 = axes[1]
        ax2.plot(points[cond], dens[cond] - prodens[cond], 'ko', linestyle='-')
        ax2.set_xticks(ticks=np.linspace(xmin, xmax, 5))
        ax2.set_ylabel('True - Approx')
        ax2.set_xlabel('Distance from the nucleus')
        ymin, ymax = np.min(dens[cond] - prodens[cond]), np.max(dens[cond] - prodens[cond])
        ax2.set_yticks(ticks=np.linspace(ymin, ymax, 5))
        if np.any(points[cond] < 1.0):
            ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        # Hide the right and top spines
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.grid(True, zorder=0, color='0.60')

        plt.tight_layout()
        plt.savefig(atom_name + "_magnified_regions" + str(i) + ".jpeg")
        #plt.show()
        plt.close()
if __name__ == "__main__":
    LIST_OF_ATOMS = [ "he", "li", "be", "b", "c", "n", "o", "f", "ne"]
    NAMES = ["Helium", "Lithium", "Beryllium","Boron", "Carbon", "Nitrogen", "Oxygen", "Fluoride", "Neon"]
    ATOMIC_NUMBER_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    USE_FILLED_VALUES_TO_ZERO = True

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    #PLOT THE ERRORS


    import os
    import horton
    from mbis_total_density import TotalMBIS
    import numpy as np

    for i, atom_name in enumerate(LIST_OF_ATOMS):
        atomic_number = ATOMIC_NUMBER_LIST[i]
        print(atomic_number, atom_name)
        file_path = os.path.expanduser('~') + r"/PythonProjects/fitting/fitting/data/examples" + "/" + atom_name

        # Create Grid Object
        rtf = horton.ExpRTransform(1.0e-30, 25, 1000)
        from fitting.density.radial_grid import Horton_Grid
        radial_grid = Horton_Grid(1e-80, 25, 1000, filled=USE_FILLED_VALUES_TO_ZERO)

        from fitting.density import Atomic_Density
        atomic_density = Atomic_Density(file_path, radial_grid.radii)

        mbis = TotalMBIS(atom_name, atomic_number, radial_grid, atomic_density.electron_density)
        storage_errors = np.load(atom_name + "_greedy_mbis_errors_iteration.npy")
        storage_errors = np.load(atom_name + "_greedy_mbis_parameters_iteration.npy")
        print(storage_errors)
        print(storage_errors)
        break

    """
    # Plotting the split regions

        parameters = np.load(atom_name + "_mbis_parameters.npy")
        coeffs, exps = parameters[:len(parameters)//2], parameters[len(parameters)//2:]
        model = mbis.get_normalized_gaussian_density(coeffs, exps)
        plot_density_sections(model, np.ravel(mbis.electron_density), mbis.grid_obj.radii, atom_name, title="Magnified "
                                                                                                 "Regions Between "
                                                                                                 "True and "
                                                                                                 "Model Densities")
    # PLOTTING ERRORS

        plt.rc('text', usetex=True)
        plt.rc('font', family='Helvetica')
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(12, 14))
        ax1.set_title(r"Various Error Measurements on the Fitting Electron Density of " + NAMES[i])
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.set_ylabel(r"$|N - \int \rho^o(r) 4 \pi r^2 dr|$")
        ax1.semilogy(np.abs(atomic_number - np.asarray(storage_of_errors[0][1:], dtype=float)), color=tableau20[0],
                     lw =2)
        ax1.grid(color=tableau20[-2], linewidth=0.5)


        ax2.set_ylabel(r"$\int |\rho(r) - \rho^o(r)| dr$")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.semilogy(storage_of_errors[1][1:], color=tableau20[0], lw=2)
        ax2.grid(color=tableau20[-2], linewidth=0.5)

        ax3.set_ylabel(r"$\int |\rho(r) - \rho^o(r) | r^2 dr$")
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.spines["bottom"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        ax3.semilogy(storage_of_errors[2][1:], color=tableau20[0], lw=2)
        ax3.grid(color=tableau20[-2], linewidth=0.5)

        ax4.set_ylabel(r"\int \rho(r) ln(\frac{\rho(r}{\rho^o(r)}) 4 \pi r^2 dr", fontsize=10)
        ax4.spines["right"].set_visible(False)
        ax4.spines["top"].set_visible(False)
        ax4.spines["bottom"].set_visible(False)
        ax4.spines["left"].set_visible(False)
        ax4.semilogy(storage_of_errors[3][1:], color=tableau20[0], lw=2)
        ax4.set_xlabel(r"Number of Iterations")
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        ax4.grid(color=tableau20[-2], linewidth=0.5)
        f.subplots_adjust(hspace=0.2)
        plt.savefig(atom_name + "_model_error.jpeg")
        plt.show()


    #PLOTTING ELECTRON DENSITIES



    import os
    import horton
    from mbis_total_density import TotalMBIS
    import numpy as np
    for i, atom_name in enumerate(LIST_OF_ATOMS):
        atomic_number = ATOMIC_NUMBER_LIST[i]
        print(atomic_number, atom_name)
        file_path = os.path.expanduser('~') + r"/PythonProjects/fitting/fitting/data/examples" + "/" + atom_name

        # Create Grid Object
        rtf = horton.ExpRTransform(1.0e-30, 25, 1000)
        from fitting.density.radial_grid import Horton_Grid
        radial_grid = Horton_Grid(1e-80, 25, 1000, filled=USE_FILLED_VALUES_TO_ZERO)

        from fitting.density import Atomic_Density
        atomic_density = Atomic_Density(file_path, radial_grid.radii)

        mbis = TotalMBIS(atom_name, atomic_number, radial_grid, atomic_density.electron_density)

        parameters = np.load(atom_name + "_mbis_parameters.npy")
        coeffs, exps = parameters[:len(parameters)//2], parameters[len(parameters)//2:]
        model = mbis.get_normalized_gaussian_density(coeffs, exps)

        print(radial_grid.integrate(np.ravel(model)))
        print(np.sum(parameters[:len(parameters)//2]))
        plt.figure(figsize=(12, 14))
        plt.rc('text', usetex=True)
        plt.rc('font', family='Helvetica')

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 9)
        ax.set_ylim(ymin=1e-10, ymax=20)
        ax.semilogy(np.ravel(radial_grid.radii)* 0.5291772082999999, model, 'r', lw=1.5, color=tableau20[0],
                     label=r"Slater Electron Density")
        ax.semilogy(np.ravel(radial_grid.radii)* 0.5291772082999999, np.ravel(atomic_density.electron_density), 'b',
                     lw=1.5, color=tableau20[6], label=r"Gaussian Fitted Electron Density")
        plt.title(r"Electron Densities of " + NAMES[i], fontsize=16)

        ax.set_xlabel(r"Distance from the nucleus [A]", fontsize=13)
        ax.set_ylabel(r"$\log[\rho(r_{bohr}^{-3})]$", fontsize=16)
        plt.grid(color=tableau20[-2], linewidth=0.5)
        plt.legend()
        plt.savefig(atom_name + "_model_results.jpeg")
        plt.show()
    """





