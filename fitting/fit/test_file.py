import numpy as np;
from fitting.density.radial_grid import *
from fitting.fit.least_squares import *
import os

#TODO ADD D1 AND D2 in the graph
def graph_and_save_the_results(self, directory):
    import os
    import zipfile
    counter = 0;
    for file in os.listdir(directory):
        print(file)
        zip_file_object = zipfile.ZipFile(file + ".zip", mode='w')

        density_model_object = GaussianTotalBasisSet(file, np.copy(self.grid), directory + "/" + file)
        fitting_object = Fitting(density_model_object)
        file_object = open(file + ".txt", mode='w')

        row_grid_points = np.copy(np.ravel(self.grid))
        # UGBS Optimize Only Coefficients Using NNLS
        UGBS_exponents = np.copy(density_model_object.UGBS_s_exponents)
        length_of_UGBS_exponents = UGBS_exponents.shape[0]
        cofactor_matrix = density_model_object.create_cofactor_matrix(UGBS_exponents)
        optimized_coefficients = fitting_object.optimize_using_nnls(cofactor_matrix)
        parameters = np.append(optimized_coefficients, np.array(UGBS_exponents))
        inte_diff = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density,
                                                                                    density_model_object.create_model(parameters, length_of_UGBS_exponents))
        diff_inte = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density,
                                                                                    density_model_object.create_model(parameters, length_of_UGBS_exponents))
        model = density_model_object.create_model(parameters, length_of_UGBS_exponents)

        file_object.write("UGBS - Optimize Coefficients Only (NNLS) \n\n")
        file_object.write("UGBS Exponents \n")
        file_object.write(str(UGBS_exponents) + "\n\n")
        file_object.write("Optimized Coefficients \n")
        file_object.write(str(optimized_coefficients) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(density_model_object.create_model(parameters, length_of_UGBS_exponents))) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff) + "\n\n")
        file_object.write("\n\n")

        # Different Gaussian Exponents - Optimize Coeff Using NNLS p = 1.25
        p = 1.25
        new_gaussian_exponents_1_25 = self.generation_of_UGBS_exponents(p, self.UGBS_s_exponents)
        print(np.shape(new_gaussian_exponents_1_25)[0])
        cofactor_matrix = self.create_cofactor_matrix(new_gaussian_exponents_1_25)
        optimized_coefficients_1_25 = fitting_object.optimize_using_nnls(cofactor_matrix)
        length_of_gauss_exponents_1_25 = np.shape(new_gaussian_exponents_1_25)[0]
        parameters_1_25 = np.append(optimized_coefficients_1_25, new_gaussian_exponents_1_25)
        model_1_25 = density_model_object.create_model(parameters_1_25, length_of_gauss_exponents_1_25)
        diff_inte_1_25 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_1_25)
        inte_diff_1_25 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_1_25)

        file_object.write("Generated Gaussian Exponents - p=1.25 - Optimize Coefficients Only (NNLS) \n\n")
        file_object.write("Generated Gaussian Exponents \n")
        file_object.write(str(new_gaussian_exponents_1_25) + "\n\n")
        file_object.write("Optimized Coefficients \n")
        file_object.write(str(optimized_coefficients_1_25) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_1_25)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_1_25) + "\n\n")
        file_object.write("\n\n")


        # Different Gaussian Exponents - Optimize Coeff Using NNLS p = 1.5
        p = 1.5
        new_gaussian_exponents_1_5 = self.generation_of_UGBS_exponents(p, self.UGBS_s_exponents)
        cofactor_matrix = self.create_cofactor_matrix(new_gaussian_exponents_1_5)
        optimized_coefficients_1_5 = fitting_object.optimize_using_nnls(cofactor_matrix)
        length_of_gauss_exponents_1_5 = np.shape(optimized_coefficients_1_5)[0]
        parameters_1_5 = np.append(optimized_coefficients_1_5, new_gaussian_exponents_1_5)
        model_1_5 = density_model_object.create_model(parameters_1_5, length_of_gauss_exponents_1_5)
        diff_inte = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_1_5)
        inte_diff_1_5 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_1_5)

        file_object.write("Different Gaussian Exponents - p=1.5 - Optimize Coefficients Only (NNLS) \n\n")
        file_object.write("Gaussian Exponents \n")
        file_object.write(str(new_gaussian_exponents_1_5) + "\n\n")
        file_object.write("Optimized Coefficients \n")
        file_object.write(str(optimized_coefficients_1_5) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_1_5)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_1_5) + "\n\n")
        file_object.write("\n\n")


        # Different Gaussian Exponents - Optimize Coeff Using NNLS p = 1.75
        p = 1.75
        new_gaussian_exponents_1_75 = self.generation_of_UGBS_exponents(p, self.UGBS_s_exponents)
        cofactor_matrix = self.create_cofactor_matrix(new_gaussian_exponents_1_75)
        optimized_coefficients_1_75 = fitting_object.optimize_using_nnls(cofactor_matrix)
        length_of_gauss_exponents_1_75 = np.shape(optimized_coefficients_1_75)[0]
        parameters_1_75 = np.append(optimized_coefficients_1_75, new_gaussian_exponents_1_75)
        model_1_75 = density_model_object.create_model(parameters_1_75, length_of_gauss_exponents_1_75)
        diff_inte_1_75 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_1_75)
        inte_diff_1_75 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_1_75)

        file_object.write("Different Gaussian Exponents - p=1.75 - Optimize Coefficients Only (NNLS) \n\n")
        file_object.write("Gaussian Exponents \n")
        file_object.write(str(new_gaussian_exponents_1_75) + "\n\n")
        file_object.write("Optimized Coefficients \n")
        file_object.write(str(optimized_coefficients_1_75) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_1_75)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_1_75) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_1_75) + "\n\n")
        file_object.write("\n\n")

        figure_name = file + " - Optimize Coeff (NNLS).png"
        title = str(counter) + "_" + file + " Density, " + "\n d=Integrate(|True - Approx| Densities) " \
                                                                                      ", Num Of Functions: " + str((length_of_gauss_exponents_1_75))
        dens_list = [(density_model_object.electron_density,"True Den"), (model_1_75,"NNLS - p=1.75 - d="  + str(inte_diff_1_75)),
                     (model_1_5, "NNLS - p=1.5 - d=" + str(inte_diff_1_5)), (model_1_25, "NNLS - p=1.25 - d=" + str(inte_diff_1_25)),
                      (model, "NNLS - UGBS - d=" + str(inte_diff))]
        density_model_object.plot_atomic_density(radial_grid=np.copy(row_grid_points), density_list=dens_list, title=title,figure_name=figure_name)
        zip_file_object.write(figure_name)
        os.remove(figure_name)




        # UGBS Optimize Only Coefficients Using SLSQP
        initial_guess_coeff = [1.0 for x in range(0, length_of_UGBS_exponents)]
        optimized_coefficients = fitting_object.optimize_using_slsqp(initial_guess_coeff, length_of_UGBS_exponents, UGBS_exponents, [] ,False, True, False)
        parameters = np.append(optimized_coefficients, UGBS_exponents)
        model_coeff_slsqp = density_model_object.create_model(parameters, length_of_UGBS_exponents)
        diff_inte_coeff_slsqp = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_coeff_slsqp)
        inte_diff_coeff_slsqp = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_coeff_slsqp)

        file_object.write("UGBS - Optimize Coefficients Only (SLSQP) \n\n")
        file_object.write("Inital Guess For Coefficients \n")
        file_object.write(str([x for x in initial_guess_coeff]) + "\n\n")
        file_object.write("UGBS Exponents \n")
        file_object.write(str(UGBS_exponents) + "\n\n")
        file_object.write("Optimized Coefficients \n")
        file_object.write(str(optimized_coefficients) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_coeff_slsqp)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_coeff_slsqp) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_coeff_slsqp) + "\n\n")
        file_object.write("\n\n")



        # Generated Gaussian Exponents - p=1.25 - Optimize Coefficients (NNLS) Then Optimize Both Using SLSQP
        initial_guess_both = np.copy(parameters_1_25)
        optimized_parameters = fitting_object.optimize_using_slsqp(initial_guess_both, length_of_gauss_exponents_1_25, [], [], True, False, False)
        coeff = np.copy(optimized_parameters[:length_of_UGBS_exponents])
        exponent = np.copy(optimized_parameters[length_of_UGBS_exponents:])
        model_1_25_nnls_slsqp = density_model_object.create_model(optimized_parameters, length_of_gauss_exponents_1_25)
        diff_inte_nnls_slsqp_1_25 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_1_25_nnls_slsqp)
        inte_diff_nnls_slsqp_1_25 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_1_25_nnls_slsqp)

        file_object.write("Generated Gaussian Exponents - p=1.25 - Optimize Coefficients (NNLS) Then Optimize Both using SLSQP \n\n")
        file_object.write("Initial Guess For Exponents (Generated Gaussian Exponents) \n")
        file_object.write(str(new_gaussian_exponents_1_25) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients_1_25) + "\n\n")
        file_object.write("Optimized Coefficients (SLSQP) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (SLSQP) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_1_25_nnls_slsqp)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_slsqp_1_25) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_slsqp_1_25) + "\n\n")
        file_object.write("\n\n")


        # Generated Gaussian Exponents - p=1.5 - Optimize Coefficients (NNLS) Then Optimize Both Using SLSQP
        initial_guess_both = np.copy(parameters_1_5)
        optimized_parameters = fitting_object.optimize_using_slsqp(initial_guess_both, length_of_gauss_exponents_1_5, [], [], True, False, False)
        coeff = np.copy(optimized_parameters[:length_of_UGBS_exponents])
        exponent = np.copy(optimized_parameters[length_of_UGBS_exponents:])
        model_nnls_slsqp_1_5 = density_model_object.create_model(optimized_parameters, length_of_gauss_exponents_1_5)
        diff_inte_nnls_slsqp_1_5 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_slsqp_1_5)
        inte_diff_nnls_slsqp_1_5 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_slsqp_1_5)

        file_object.write("Generated Gaussian Exponents - p=1.5 -  Optimize Coefficients (NNLS) Then Optimize Both using SLSQP \n\n")
        file_object.write("Initial Guess For Exponents (Generated Gaussian Exponents) \n")
        file_object.write(str(new_gaussian_exponents_1_5) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients_1_5) + "\n\n")
        file_object.write("Optimized Coefficients (SLSQP) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (SLSQP) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_slsqp_1_5)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_slsqp_1_5) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_slsqp_1_5) + "\n\n")
        file_object.write("\n\n")


        # Generated Gaussian Exponents - p=1.75 - Optimize Coefficients (NNLS) Then Optimize Both Using SLSQP
        initial_guess_both = np.copy(parameters_1_75)
        optimized_parameters = fitting_object.optimize_using_slsqp(initial_guess_both, length_of_gauss_exponents_1_75, [], [], True, False, False)
        coeff = np.copy(optimized_parameters[:length_of_UGBS_exponents])
        exponent = np.copy(optimized_parameters[length_of_UGBS_exponents:])
        model_nnls_slsqp_1_75 = density_model_object.create_model(optimized_parameters, length_of_gauss_exponents_1_75)
        diff_inte_nnls_slsqp_1_75 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_slsqp_1_75)
        inte_diff_nnls_slsqp_1_75 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_slsqp_1_75)

        file_object.write("Generated Gaussian Exponents - p=1.75 - Optimize Coefficients (NNLS) Then Optimize Both using SLSQP \n\n")
        file_object.write("Initial Guess For Exponents (Generated Gaussian Exponents) \n")
        file_object.write(str(new_gaussian_exponents_1_75) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients_1_75) + "\n\n")
        file_object.write("Optimized Coefficients (SLSQP) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (SLSQP) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_slsqp_1_75)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_slsqp_1_75) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_slsqp_1_75) + "\n\n")
        file_object.write("\n\n")


        # UGBS Optimize Coefficients (NNLS) Then Optimize Exponents Using SLSQP
        cofactor_matrix = density_model_object.create_cofactor_matrix(UGBS_exponents)
        optimized_coefficients = fitting_object.optimize_using_nnls(cofactor_matrix)

        optimized_exponents = fitting_object.optimize_using_slsqp(UGBS_exponents, length_of_UGBS_exponents,[], optimized_coefficients, False, False, True)
        parameters = np.append(optimized_coefficients, optimized_exponents)
        model_nnls_slsqp_ugbs = density_model_object.create_model(parameters, length_of_UGBS_exponents)
        diff_inte_nnls_slsqp_ugbs = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_slsqp_ugbs)
        inte_diff_nnls_slsqp_ugbs = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_slsqp_ugbs)

        file_object.write("UGBS - Optimize Coefficients (NNLS) Then Optimize Exponents Using SLSQP \n\n")
        file_object.write("Initial Guess for Exponents\\UGBS Exponents \n")
        file_object.write(str(UGBS_exponents) + "\n\n")
        file_object.write("Optimized Coefficients (NNLS) \n")
        file_object.write(str(optimized_coefficients) + "\n\n")
        file_object.write("Optimized Exponents (SLSQP) \n")
        file_object.write(str(optimized_exponents) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_slsqp_ugbs)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_slsqp_ugbs) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_slsqp_ugbs) + "\n\n")
        file_object.write("\n\n")


        # UGBS - Optimize Coefficients (NNLS) Then Optimize Both using SLSQP
        initial_guess_both = np.append(optimized_coefficients, UGBS_exponents)
        optimized_parameters = fitting_object.optimize_using_slsqp(initial_guess_both, length_of_UGBS_exponents, [], [], True, False, False)
        coeff = optimized_parameters[:length_of_UGBS_exponents]; exponent = optimized_parameters[length_of_UGBS_exponents:]
        model_nnls_slsqp = density_model_object.create_model(optimized_parameters, length_of_UGBS_exponents)
        diff_inte_nnls_slsqp = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_slsqp)
        inte_diff_nnls_slsqp = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_slsqp)

        file_object.write("UGBS - Optimize Coefficients (NNLS) Then Optimize Both using SLSQP \n\n")
        file_object.write("Initial Guess for Exponents \n")
        file_object.write(str(UGBS_exponents) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients) + "\n\n")
        file_object.write("Optimized Coefficients (SLSQP) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (SLSQP) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_slsqp)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff) + "\n\n")
        file_object.write("\n\n")

        figure_name = file + " SLSQP"
        title = str(counter) + "_" + file + " Density, " + "\n d=Integrate(|True - Approx| Densities) " \
                                                                                             ", Num Of Functions: " + str((length_of_UGBS_exponents))
        dens_list_slsqp = [(np.copy(density_model_object.electron_density), "True Density"), (model_coeff_slsqp, "SLSQP - UGBS - d=" + "{0:.2f}".format(round(inte_diff_coeff_slsqp, 2))),
                           (model_1_25_nnls_slsqp, "NNLS/SLSQP(Both) - p=1.25 - d=" + "{0:.2f}".format(round(inte_diff_nnls_slsqp_1_25, 2))),
                           (model_nnls_slsqp_1_5, "NNLS/SLSQP(Both) - p=1.5 - d=" + "{0:.2f}".format(round(inte_diff_nnls_slsqp_1_5, 2))),
                           (model_nnls_slsqp_1_75, "NNLS/SLSQP(Both) - p=1.75 - d=" + "{0:.2f}".format(round(inte_diff_nnls_slsqp_1_75, 2))),
                           (model_nnls_slsqp, "NNLS/SLSQP(Both) - UGBS - d=" + "{0:.2f}".format(round(inte_diff_nnls_slsqp, 2))),
                           (model_nnls_slsqp_ugbs, "NNLS/SLSQP (Expon) - UGBS - d=" + "{0:.2f}".format(round(inte_diff_nnls_slsqp_ugbs, 2)))]
        density_model_object.plot_atomic_density(radial_grid=np.copy(row_grid_points), density_list=dens_list_slsqp, title=title,figure_name=figure_name)
        zip_file_object.write(figure_name + ".png")
        os.remove(figure_name + ".png")







        # Generated Gaussian Exponents - p=1.25 - Optimize Coefficients (NNLS) Then Optimize Both Using BFGS
        initial_guess_both = np.copy(parameters_1_25)
        optimized_parameters = fitting_object.optimize_using_l_bfgs(initial_guess_both, length_of_gauss_exponents_1_25, [], [], True, False, False)
        coeff = np.copy(optimized_parameters[:length_of_UGBS_exponents])
        exponent = np.copy(optimized_parameters[length_of_UGBS_exponents:])
        model_nnls_bfgs_1_25 = density_model_object.create_model(optimized_parameters, length_of_gauss_exponents_1_25)
        diff_inte_nnls_bfgs_1_25 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_bfgs_1_25)
        inte_diff_nnls_bfgs_1_25 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_bfgs_1_25)

        file_object.write("Generated Gaussian Exponents - p=1.25 - Optimize Coefficients (NNLS) Then Optimize Both using BFGS \n\n")
        file_object.write("Initial Guess For Exponents (Generated Gaussian Exponents) \n")
        file_object.write(str(new_gaussian_exponents_1_25) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients_1_25) + "\n\n")
        file_object.write("Optimized Coefficients (BFGS) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (BFGS) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_bfgs_1_25)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_bfgs_1_25) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_bfgs_1_25) + "\n\n")
        file_object.write("\n\n")



        # Generated Gaussian Exponents - p=1.5 - Optimize Coefficients (NNLS) Then Optimize Both Using BFGS
        initial_guess_both = np.copy(parameters_1_5)
        optimized_parameters = fitting_object.optimize_using_l_bfgs(initial_guess_both, length_of_gauss_exponents_1_5, [], [], True, False, False)
        coeff = np.copy(optimized_parameters[:length_of_UGBS_exponents])
        exponent = np.copy(optimized_parameters[length_of_UGBS_exponents:])
        model_nnls_bfgs_1_5 = density_model_object.create_model(optimized_parameters, length_of_gauss_exponents_1_5)
        diff_inte_nnls_bfgs_1_5 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_bfgs_1_5)
        inte_diff_nnls_bfgs_1_5 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_bfgs_1_5)

        file_object.write("Generated Gaussian Exponents - p=1.5 -  Optimize Coefficients (NNLS) Then Optimize Both using BFGS \n\n")
        file_object.write("Initial Guess For Exponents (Generated Gaussian Exponents) \n")
        file_object.write(str(new_gaussian_exponents_1_5) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients_1_5) + "\n\n")
        file_object.write("Optimized Coefficients (BFGS) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (BFGS) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_bfgs_1_5)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_bfgs_1_5) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_bfgs_1_5) + "\n\n")
        file_object.write("\n\n")


        # Generated Gaussian Exponents - p=1.75 - Optimize Coefficients (NNLS) Then Optimize Both Using BFGS
        initial_guess_both = np.copy(parameters_1_75)
        optimized_parameters = fitting_object.optimize_using_l_bfgs(initial_guess_both, length_of_gauss_exponents_1_75, [], [], True, False, False)
        coeff = np.copy(optimized_parameters[:length_of_UGBS_exponents])
        exponent = np.copy(optimized_parameters[length_of_UGBS_exponents:])
        model_nnls_bfgs_1_75 = density_model_object.create_model(optimized_parameters, length_of_gauss_exponents_1_75)
        diff_inte_nnls_bfgs_1_75 = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_bfgs_1_75)
        inte_diff_nnls_bfgs_1_75 = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_bfgs_1_75)

        file_object.write("Generated Gaussian Exponents - p=1.75 - Optimize Coefficients (NNLS) Then Optimize Both using BFGS \n\n")
        file_object.write("Initial Guess For Exponents (Generated Gaussian Exponents) \n")
        file_object.write(str(new_gaussian_exponents_1_75) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients_1_75) + "\n\n")
        file_object.write("Optimized Coefficients (BFGS) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (BFGS) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_bfgs_1_75)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_bfgs_1_75) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_bfgs_1_75) + "\n\n")
        file_object.write("\n\n")



        # UGBS Optimize Only Coefficients Using L_FMIN_BFGS
        initial_guess_coeff = [1.0 for x in range(0, length_of_UGBS_exponents)]
        optimized_coefficients = fitting_object.optimize_using_l_bfgs(initial_guess_coeff, length_of_UGBS_exponents, UGBS_exponents, [] ,False, True, False)
        parameters = np.append(optimized_coefficients, UGBS_exponents)
        model_coeff_bfgs = density_model_object.create_model(parameters, length_of_UGBS_exponents)
        diff_inte_coeff_bfgs = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_coeff_bfgs)
        inte_diff_coeff_bfgs = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_coeff_bfgs)

        file_object.write("UGBS - Optimize Coefficients Only (BFGS) \n\n")
        file_object.write("Inital Guess For Coefficients \n")
        file_object.write(str([x for x in initial_guess_coeff]) + "\n\n")
        file_object.write("UGBS Exponents \n")
        file_object.write(str(UGBS_exponents) + "\n\n")
        file_object.write("Optimized Coefficients \n")
        file_object.write(str(optimized_coefficients) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_coeff_bfgs)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_coeff_bfgs) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_coeff_bfgs) + "\n\n")
        file_object.write("\n\n")



        # UGBS Optimize Coefficients (NNLS) Then Optimize Exponents Using l_fmin_bfgs
        optimized_exponents = fitting_object.optimize_using_l_bfgs(UGBS_exponents, length_of_UGBS_exponents, [], optimized_coefficients, False, False, True)
        parameters = np.append(optimized_coefficients, optimized_exponents)
        model_nnls_bfgs_ugbs = density_model_object.create_model(parameters, length_of_UGBS_exponents)
        diff_inte_nnls_bfgs_ugbs = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_bfgs_ugbs)
        inte_diff_nnls_bfgs_ugbs = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_bfgs_ugbs)

        file_object.write("UGBS - Optimize Coefficients (NNLS) Then Optimize Exponents Using BFGS \n\n")
        file_object.write("Initial Guess for Exponents\\UGBS Exponents \n")
        file_object.write(str(UGBS_exponents) + "\n\n")
        file_object.write("Optimized Coefficients (NNLS) \n")
        file_object.write(str(optimized_coefficients) + "\n\n")
        file_object.write("Optimized Exponents (BFGS) \n")
        file_object.write(str(optimized_exponents) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_bfgs_ugbs)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_bfgs_ugbs) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_bfgs_ugbs) + "\n\n")
        file_object.write("\n\n")


        #UGBS - Optimize Coefficients (NNLS) Then Optimize Both Using BFGS
        initial_guess_both = np.append(optimized_coefficients, UGBS_exponents)
        optimized_parameters = fitting_object.optimize_using_l_bfgs(initial_guess_both, length_of_UGBS_exponents, [], [], True, False, False)
        coeff = optimized_parameters[:length_of_UGBS_exponents]; exponent = optimized_parameters[length_of_UGBS_exponents:]
        model_nnls_bfgs_both = density_model_object.create_model(optimized_parameters, length_of_UGBS_exponents)
        diff_inte_nnls_bfgs_both = density_model_object.measure_error_by_difference_of_integration(density_model_object.electron_density, model_nnls_bfgs_both)
        inte_diff_nnls_bfgs_both = density_model_object.measure_error_by_integration_of_difference(density_model_object.electron_density, model_nnls_bfgs_both)

        file_object.write("UGBS - Optimize Coefficients (NNLS) Then Optimize Both (BFGS) \n\n")
        file_object.write("Initial Guess for Exponents \n")
        file_object.write(str(UGBS_exponents) + "\n\n")
        file_object.write("Initial Guess for Coefficients \n")
        file_object.write(str(optimized_coefficients) + "\n\n")
        file_object.write("Optimized Coefficients (SLSQP) \n")
        file_object.write(str(coeff) + "\n\n")
        file_object.write("Optimized Exponents (SLSQP) \n")
        file_object.write(str(exponent) + "\n\n")
        file_object.write("Integration of Electron Density Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrated_total_electron_density) + "\n\n")
        file_object.write("Integration of Model Times Grid Squared Using Trapz:\n")
        file_object.write(str(density_model_object.integrate_model_using_trapz(model_nnls_bfgs_both)) + "\n\n")
        file_object.write("Error=|Integrate(True) - Integrate(Approx)| Densities\n")
        file_object.write(str(diff_inte_nnls_bfgs_both) + "\n\n")
        file_object.write("Error=Integrate(|True - Approx|) Densities\n")
        file_object.write(str(inte_diff_nnls_bfgs_both) + "\n\n")
        file_object.write("\n\n")

        figure_name = file + " BFGS"
        title = str(counter) + "_" + file + " Density, " + "\n d=Integrate(|True - Approx| Densities) " \
                                                                                             ", Num Of Functions: " + str((length_of_UGBS_exponents))
        dens_list_bfgs = [(np.copy(density_model_object.electron_density),"True Density"),
                          (model_coeff_bfgs, "BFGS - UGBS - d=" + "{0:.2f}".format(round(inte_diff_coeff_bfgs, 3))),
                          (model_nnls_bfgs_1_25, "NNLS/BFGS(Both) - p=1.25 - d=" + "{0:.4f}".format(round(inte_diff_nnls_bfgs_1_25, 4))),
                          (model_nnls_bfgs_1_5, "NNLS/BFGS(Both) - p=1.5 - d=" + "{0:.4f}".format(round(inte_diff_nnls_bfgs_1_5, 4))),
                          (model_nnls_bfgs_1_75, "NNLS/BFGS(Both) - p=1.75 - d=" + "{0:.2f}".format(round(inte_diff_nnls_bfgs_1_75, 3))),
                          (model_nnls_bfgs_both, "NNLS/BFGS (Both) - UGBS - d=" + "{0:.2f}".format(round(inte_diff_nnls_bfgs_both, 3))),
                          (model_nnls_bfgs_ugbs, "NNLS/BFGS (Expo) - UGBS - d=" + "{0:.2f}".format(round(inte_diff_nnls_bfgs_ugbs, 3)))]
        density_model_object.plot_atomic_density(radial_grid=np.copy(row_grid_points), density_list=dens_list_bfgs, title=title,figure_name=figure_name)
        zip_file_object.write(figure_name + ".png")
        os.remove(figure_name + ".png")



        zip_file_object.write(file + ".txt", arcname=file + ".txt")

        counter +=1;
        zip_file_object.close()
        file_object.close()
        os.remove(file + ".txt")
        break;

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