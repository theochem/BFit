def plot_atomic_density(radial_grid, model, electron_density, title, figure_name):
    # Density List should be in the form
    # [(electron density, legend reference),(model1, legend reference), ..]
    import matplotlib.pyplot as plt
    colors = ["#FF00FF", "#FF0000", "#FFAA00", "#00AA00", "#00AAFF", "#0000FF", "#777777", "#00AA00", "#00AAFF"]
    ls_list = ['-', ':', ':', '-.', '-.', '--', '--', ':', ':']

    radial_grid *= 0.5291772082999999  # convert a.u. to angstrom
    plt.semilogy(radial_grid, model, lw=3, label="approx_model", color=colors[0], ls=ls_list[0])
    plt.semilogy(radial_grid, electron_density, lw=3, label="True Model", color=colors[2], ls=ls_list[3])
    # plt.xlim(0, 25.0*0.5291772082999999)
    plt.xlim(0, 9)
    plt.ylim(ymin=1e-9)
    plt.xlabel('Distance from the nucleus [A]')
    plt.ylabel('Log(density [Bohr**-3])')
    plt.title(title)
    plt.legend()
    plt.savefig(figure_name)
    plt.show()
    plt.close()

def plot_density_sections(dens, prodens, points, title='None'):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib import rcParams
    # choose fonts
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']

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
    for cond in conditions:
        # setup figure
        fig, axes = plt.subplots(2, 1)
        fig.suptitle(title, fontsize=12, fontweight='bold')

        # plot true & model density
        ax1 = axes[0]
        ax1.plot(points[cond], dens[cond], 'ro', linestyle='-', label='True')
        ax1.plot(points[cond], prodens[cond], 'bo', linestyle='--', label='Approx')
        ax1.legend(loc=0, frameon=False)
        xmin, xmax = np.min(points[cond]), np.max(points[cond])
        ax1.set_xticks(ticks=np.linspace(xmin, xmax, 5))
        ymin, ymax = np.min(dens[cond]), np.max(dens[cond])
        ax1.set_yticks(ticks=np.linspace(ymin, ymax, 5))
        ax1.set_ylabel('Density')
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
        ax2.set_xlabel('Distance from the nueleus')
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
        plt.show()
        plt.close()



def plot_density_sections2(dens, prodens, points, atom_name, title='None'):
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
    colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44)]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)
    for i, cond in enumerate(conditions):
        # setup figure
        fig, axes = plt.subplots(2, 1)

        # plot true & model density
        ax1 = axes[0]
        ax1.set_title(title, fontsize=12, fontweight='bold')
        if type(dens) is list:
            for j, d in enumerate(dens):
                ax1.plot(points[cond], d[cond], 'o', color=colors[j], linestyle='-', label=str(j + 26) +
                                                                                           " Basis Functions")
        else:
            ax1.plot(points[cond], dens[cond], 'ro', linestyle='-', label=r'True')
        ax1.plot(points[cond], prodens[cond], 'bo', linestyle='--', label=r'Approx')
        ax1.legend(loc=0, frameon=False)
        xmin, xmax = np.min(points[cond]), np.max(points[cond])
        ax1.set_xticks(ticks=np.linspace(xmin, xmax, 5))
        if type(dens) is list:
            ymin, ymax = np.min([np.min(x[cond]) for x in dens]), np.max([np.max(x[cond]) for x in dens])
        else:
            ymin, ymax = np.min(dens[cond]), np.max(dens[cond])
        ax1.set_yticks(ticks=np.linspace(ymin, ymax, 5))
        ax1.set_ylabel(r'Densities ')
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
        if type(dens) is list:
            for j, d in enumerate(dens):
                ax2.plot(points[cond], d[cond] - prodens[cond], 'o', color=colors[j], linestyle='-')
        else:
            ax2.plot(points[cond], dens[cond] - prodens[cond], 'ko', linestyle='-')
        ax2.set_xticks(ticks=np.linspace(xmin, xmax, 5))
        ax2.set_ylabel('True - Approx')
        ax2.set_xlabel('Distance from the nucleus')
        if type(dens) is list:
            ymin, ymax = np.min(np.min([np.min(x[cond]) for x in dens]) - prodens[cond]), \
                         np.max(np.max([np.max(x[cond]) for x in dens]) - prodens[cond])
        else:
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
        plt.savefig(atom_name + "_greedy_magnified_regions" + str(i) + ".jpeg")
        #plt.show()
        plt.close()