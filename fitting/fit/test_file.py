import numpy as np;
from fitting.density.radial_grid import *
from fitting.fit.least_squares import *
import os


def plot_atomic_desnity(radial_grid, density_list, title, figure_name):
    import matplotlib.pyplot as plt
    colors = ["#FF00FF", "#FF0000", "#FFAA00", "#00AA00", "#00AAFF", "#0000FF", "#777777", "#00AA00", "#00AAFF"]
    ls_list = ['-', ':', ':', '-.', '-.', '--', '--', ':', ':']
    assert isinstance(density_list, list)
    radial_grid *= 0.5291772082999999   #convert a.u. to angstrom
    for i, item in enumerate(density_list):
        dens, label = item
        # plot with log scaling on the y axis
        plt.semilogy(radial_grid, dens, lw=3, label=label, color=colors[i], ls=ls_list[i])

    #plt.xlim(0, 25.0*0.5291772082999999)
    plt.xlim(0, 7.5)
    plt.ylim(ymin=1e-8)
    plt.xlabel('Distance from the nucleus [A]')
    plt.ylabel('Log(density [Bohr**-3])')
    plt.title(title)
    plt.legend(loc=0)
    plt.savefig(figure_name)
    plt.close()


GRID = Radial_Grid(4)
row_grid_points = radial_grid.grid_points(200, 300, [50, 75, 100])
column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

file_path_slator_files = os.path.dirname(os.path.abspath(__file__))[:-3]  + "examples\\"#remove /fit
print(file_path_slator_files)

FACTR = 1e-10
STR_FACTR = "1e-10"
MAXIMUM_EXPONENTS = 34
FACTOR = 2.0
ACCURACY = 0.01
TECHNIQUE = "SLSQP"
counter = 1;


for file in os.listdir(file_path_slator_files):
    file_object = open(file + ".txt", mode='w')
    try:
        file_path_element = (file_path_slator_files + file )
        print(file)
        atom = Model('be', file_path_element, column_grid_points)
        UGBS_exp = np.copy(atom.exponents)

        def measure_error_helper(coeff, exponents):
            model = atom.model(coeff, exponents)
            diff = np.absolute(np.ravel(atom.electron_density) - model)
            grid = np.ravel(atom.grid)
            integration = np.trapz(y=diff * np.ravel(np.power(atom.grid, 2)), x=grid)

            #inte = self.integration(coeff, exponents)
            #true_val = self.true_value(100)
            #integration = np.absolute(inte-true_val)
            return(integration)

        def difference_of_seperate_integration(coeffi, exponents2):
            model = atom.model(coeffi, exponents2)
            grid = np.copy(np.ravel(atom.grid))
            integrate_model = np.trapz(y=np.ravel(model) * np.ravel(np.power(grid, 2.0)), x=grid)
            print("model", integrate_model)
            integrate_elec_d = np.trapz(y=np.ravel(atom.electron_density) * np.ravel(np.power(grid, 2.0)), x=grid)
            print("ed",integrate_elec_d)
            diff = np.absolute(integrate_elec_d - integrate_model)
            print("Diff",diff)
            return(diff)

        #UGBS OPTIMIZE COEFFICIENTS FIRST
        #TODO Fix error on graph
        cofactor_matrix = atom.cofactor_matrix()
        coefficients = atom.nnls_coefficients(cofactor_matrix)
        model2 = atom.model(coefficients, atom.exponents)
        error = measure_error_helper(coefficients, atom.exponents)
        error2 = difference_of_seperate_integration(coefficients, atom.exponents)
        integrate = np.trapz(np.ravel(atom.grid**2) * np.ravel(atom.electron_density), np.ravel(atom.grid))

        file_object.write("UGBS - Optimize Coefficients Only (NNLS) \n\n")
        file_object.write("UGBS Exponents \n")
        file_object.write(str(atom.exponents) + "\n\n")
        file_object.write("Optimized Coefficients \n")
        file_object.write(str(coefficients) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(integrate) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(atom.integration(coefficients, atom.exponents)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(error2) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(error) + "\n\n")
        file_object.write("\n\n")


        figure_name = file + " UGBS - Coeffs"
        title = str(counter) + "_" + file + " Density, " + "Accuracy Desired: " + str(ACCURACY) + "\n d=Integrate(|True - Approx| Densities) " \
                                                                                                 ", Num Of Functions: " + str(len(coefficients))
        plot_atomic_desnity(radial_grid=row_grid_points, density_list=[(be.electron_density,"True Den"), (model2,
                    "NNLS" + ", d=" + str(error))], title=title,figure_name=figure_name)


        #UGBS Optimize Coefficients (SLSQP)
        file_object.write("UGBS - Optimize Coefficients (SLSQP) Using NNLS Coefficients as Initial Guess  \n\n")
        #Use SLSQP to do it
        coeff = atom.f_min_slsqp_coefficients(initial_guess=coefficients,exponents=atom.exponents,change_exponents=True)
        error = measure_error_helper(coeff, atom.exponents)
        error2 = difference_of_seperate_integration(coeff, atom.exponents)

        file_object.write("Optimized Coeffs:\n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(integrate) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(atom.integration(coeff, atom.exponents)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(error2) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(error) + "\n\n")
        file_object.write("\n\n")

        #UGBS Optimize Coe
        file_object.write("UGBS - Optimize Coefficients (L BFGS) Using NNLS Coefficients as Initial GUess \n\n")
        coeff2 = atom.f_min_slsqp_coefficients(coefficients, exponents=atom.exponents, change_exponents=True, use_slsqp=False)
        error = measure_error_helper(coeff2, UGBS_exp)
        error2 = difference_of_seperate_integration(coeff2, UGBS_exp)

        file_object.write("Optimized Coeffs:\n")
        file_object.write(str(coeff2) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(integrate) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        print(np.shape(atom.grid), np.shape(row_grid_points), np.shape(atom.exponents), np.shape(UGBS_exp))
        print(str(np.trapz(np.ravel(atom.model(coeff2,UGBS_exp)) * np.ravel(np.power(atom.grid,2.0)), np.ravel(atom.grid)) ))
        file_object.write(str(np.trapz(np.ravel( np.ravel(atom.model(coeff2,UGBS_exp)) * np.ravel(np.power(atom.grid,2.0))), np.ravel(atom.grid)))+ "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(error2) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(error) + "\n\n")
        file_object.write("\n\n")

        R"""
        best_parameter, error = atom.pauls_GA(FACTOR, ACCURACY, factr=FACTR, maximum_exponents=MAXIMUM_EXPONENTS, opt_both=True)
        error = np.round(error , 2)
        #Calculate Electron Density
        size = best_parameter.shape[0]
        coeff,exp = best_parameter[0:size/2], best_parameter[size/2:]
        electron_density_calculated = atom.model(coeff, exp)

        figure_name = file + " " + TECHNIQUE
        title = str(counter) + "_" + file + " Density " + "Accuracy Desired: " + str(ACCURACY) + "\n d=Integrate(|True - Approx| Densities)"
        plot_atomic_desnity(radial_grid=row_grid_points, density_list=[(be.electron_density,"True Den"), (electron_density_calculated,
                    TECHNIQUE + ", d=" + str(error) + " factr: " + STR_FACTR +", Num Of Funcs: " + str(size/2))], title=title,figure_name=figure_name)"""
        print("\n")
        break;
    except Exception as e:
        print(file + "error")
        import traceback, os.path
        print(e)
        print("Error is ", e, 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno, "\n"))

    file_object.close()
    counter += 1