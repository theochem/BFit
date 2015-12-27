from fitting.fit.model import *
import numpy as np


#TODO CHANGE GRID TO BEING ONE DIMENSIONAL
class GaussianTotalBasisSet(DensityModel):
    def __init__(self, element_name, grid, file_path):
        DensityModel.__init__(self, element_name, grid, file_path)

    def create_model(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        assert parameters.ndim == 1
        def check_dimension_and_shape():
            assert exponents.ndim == 1
            assert np.shape(exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(exponential)[1] == np.shape(exponents)[0]
            assert exponential.ndim == 2
            assert coefficients.ndim == 1
            assert (coefficients.shape)[0] == exponential.shape[1]
            assert gaussian_density.ndim == 1
            assert np.shape(gaussian_density)[0] == np.shape(self.grid)[0]

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return(gaussian_density)

    def cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianTotalBasisSet, self).calculate_residual(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianTotalBasisSet, self).calculate_residual(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianTotalBasisSet, self).calculate_residual(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)

        residual_squared = np.power(residual, 2.0)
        return(np.sum(residual_squared))

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        def derivative_wrt_coefficients():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_coeff)

        def derivative_wrt_exponents():
                derivative_exp = []
                for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                    exponent = exponents[index]
                    g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                    derivative = -f_function * np.ravel(g_function)
                    derivative_exp.append(np.ravel(derivative))
                assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
                return(derivative_exp)

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianTotalBasisSet, self).calculate_residual(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

            f_function = 2.0 * residual
            derivative = []

            derivative_coeff = derivative_wrt_coefficients()
            derivative_exp = derivative_wrt_exponents()
            derivative = derivative + derivative_coeff
            derivative = derivative + derivative_exp

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianTotalBasisSet, self).calculate_residual(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_coefficients()

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianTotalBasisSet, self).calculate_residual(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_exponents()
        return(np.sum(derivative, axis=1))

    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)

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
            new_gaussian_exponents_1_25 = self.generation_of_UGBS_exponents(p)
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
            new_gaussian_exponents_1_5 = self.generation_of_UGBS_exponents(p)
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
            new_gaussian_exponents_1_75 = self.generation_of_UGBS_exponents(p)
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


# Grab Element and the file path
element = "be"
file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + element + ".slater"

#Create Grid for the modeling
from fitting.density.radial_grid import *
radial_grid = Radial_Grid(4)
row_grid_points = radial_grid.grid_points(200, 300, [50, 75, 100])
column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

# Create a total gaussian basis set
be = GaussianTotalBasisSet(element, column_grid_points, file_path)
fitting_object = Fitting(be)

# Save Results
"""
import os
directory = os.path.dirname(__file__).rsplit('/', 2)[0] + "/fitting/examples"
#be.graph_and_save_the_results(directory)
"""

# Fit Model Using Greedy Algorithm
#fitting_object.forward_greedy_algorithm(2.0, 0.01, np.copy(be.electron_density), maximum_num_of_functions=100)
#fitting_object.find_best_UGBS_exponents(1, p=1.5)
#fitting_object.analytically_solve_objective_function(be.electron_density, 1.0)
#be.generation_of_UGBS_exponents(1.25)


class GaussianCoreBasisSet(DensityModel):
    def __init__(self, element_symbol, grid, file_path):
        DensityModel.__init__(self, element_symbol, grid, file_path)

    def create_model(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        assert parameters.ndim == 1
        def check_dimension_and_shape():
            assert exponents.ndim == 1
            assert np.shape(exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(exponential)[1] == np.shape(exponents)[0]
            assert exponential.ndim == 2
            assert coefficients.ndim == 1
            assert (coefficients.shape)[0] == exponential.shape[1]
            assert gaussian_density.ndim == 1
            assert np.shape(gaussian_density)[0] == np.shape(self.grid)[0]

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return(gaussian_density)

    def cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)

        residual_squared = np.power(residual, 2.0)
        return(np.sum(residual_squared))

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        def derivative_wrt_coefficients():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_coeff)

        def derivative_wrt_exponents():
                derivative_exp = []
                for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                    exponent = exponents[index]
                    g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                    derivative = -f_function * np.ravel(g_function)
                    derivative_exp.append(np.ravel(derivative))
                assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
                return(derivative_exp)

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

            f_function = 2.0 * residual
            derivative = []

            derivative_coeff = derivative_wrt_coefficients()
            derivative_exp = derivative_wrt_exponents()
            derivative = derivative + derivative_coeff
            derivative = derivative + derivative_exp

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_coefficients()

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_exponents()
        return(np.sum(derivative, axis=1))

    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)

# Create a core basis set object
#be_core = GaussianCoreBasisSet("C", column_grid_points, file_path)

#Create fitting object based on core and fit the model using the greedy algorithm
#fitting_object_core = Fitting(be_core)
#fitting_object_core.forward_greedy_algorithm(2.0, 0.01, np.copy(be_core.electron_density_core),maximum_num_of_functions=100)

class GaussianValenceBasisSet(DensityModel):
    def __init__(self, element_symbol, grid, file_path):
        DensityModel.__init__(self, element_symbol, grid, file_path)

    def create_model(self, parameters, num_of_s_funcs, num_of_p_funcs, optimize_both=True, optimize_coeff=False, optimize_exp=False):
        s_coefficients = parameters[:num_of_s_funcs]
        s_exponents = parameters[num_of_s_funcs: 2 * num_of_s_funcs]

        p_coefficients = parameters[num_of_s_funcs * 2 : num_of_s_funcs * 2 + num_of_p_funcs]
        p_exponents = parameters[num_of_s_funcs * 2 + num_of_p_funcs:]

        s_exponential = np.exp(-s_exponents * np.power(self.grid, 2.0))
        p_exponential = np.exp(-p_exponents * np.power(self.grid, 2.0))

        def check_type_and_dimensions():
            assert type(s_coefficients).__module__ == np.__name__
            assert type(s_exponents).__module__ == np.__name__
            assert type(p_coefficients).__module__ == np.__name__
            assert type(p_exponents).__module__ == np.__name__

            assert s_exponents.ndim == 1; assert p_exponents.ndim == 1;
            assert np.shape(p_exponential)[1] == np.shape(p_exponents)[0]
            assert np.shape(s_exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(p_exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(s_exponential)[1] == np.shape(s_exponents)[0]

            assert s_exponential.ndim == 2; assert p_exponential.ndim == 2
            assert s_coefficients.ndim == 1
            assert (s_coefficients.shape)[0] == s_exponential.shape[1]

            assert s_gaussian_model.ndim == 1
            assert np.shape(s_gaussian_model)[0] == np.shape(self.grid)[0]

        check_type_and_dimensions()

        s_gaussian_model = np.dot(s_exponential, s_coefficients)
        p_gaussian_model = np.dot(p_exponential, p_coefficients)
        p_gaussian_model = np.ravel(p_gaussian_model)  * np.ravel(np.power(self.grid, 2.0))

        return(s_gaussian_model + p_gaussian_model)

    def cost_function(self, parameters, num_of_s_funcs, num_of_p_funcs, optimize_both=True, optimize_coeff=False, optimize_exp=False):
        assert type(parameters).__module__ == np.__name__
        assert isinstance(num_of_s_funcs, int)
        assert isinstance(num_of_p_funcs, int)

        s_coefficients = parameters[:num_of_s_funcs]
        s_exponents = parameters[num_of_s_funcs:2 * num_of_s_funcs]

        p_coefficients = parameters[num_of_s_funcs * 2 :num_of_s_funcs * 2 + num_of_p_funcs]
        p_exponents = parameters[num_of_s_funcs * 2 + num_of_p_funcs:]

        residual = super(GaussianValenceBasisSet, self).calculate_residual_based_on_valence(parameters, num_of_s_funcs, num_of_p_funcs)
        residual_squared = np.power(residual, 2.0)

        return(np.sum(residual_squared))

    def derivative_of_cost_function(self, parameters, num_of_s_funcs, num_of_p_funcs, optimize_both=True, optimize_coeff=False, optimize_exp=False):
        s_coefficients = parameters[:num_of_s_funcs]
        s_exponents = parameters[num_of_s_funcs:2 * num_of_s_funcs]

        p_coefficients = parameters[num_of_s_funcs * 2 :num_of_s_funcs * 2 + num_of_p_funcs]
        p_exponents = parameters[num_of_s_funcs * 2 + num_of_p_funcs:]

        residual = super(GaussianValenceBasisSet, self).calculate_residual_based_on_valence(parameters, num_of_s_funcs, num_of_p_funcs)
        f_function = 2.0 * residual
        derivative = []

        def derivative_coeff_helper():
            derivative_s_coeff = []
            for exp in s_exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_s_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_s_coeff[0])[0] == np.shape(self.grid)[0]
            derivative_p_coeff = []
            for exp in p_exponents:
                g_function = -1.0 * np.power(self.grid, 2.0) * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_p_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_s_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_s_coeff, derivative_p_coeff)

        def derivative_exp_helper():
            derivative_s_exp = []
            for index, coeff in np.ndenumerate(np.ravel(s_coefficients)):
                exponent = s_exponents[index]
                g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                derivative = -f_function * np.ravel(g_function)
                derivative_s_exp.append(np.ravel(derivative))
            assert np.shape(derivative_s_exp[0])[0] == np.shape(self.grid)[0]
            derivative_p_exp = []
            for index, coeff in np.ndenumerate(np.ravel(p_coefficients)):
                exponent = p_exponents[index]
                g_function = -coeff * np.power(self.grid, 4.0) * np.exp(-1.0 * exponent * self.grid**2)
                derivative = -f_function * np.ravel(g_function)
                derivative_p_exp.append(np.ravel(derivative))
            return(derivative_s_exp, derivative_p_exp)

        derivative_s_coeff, derivative_p_coeff = derivative_coeff_helper()
        derivative_s_exp, derivative_p_exp = derivative_exp_helper()
        derivative = derivative + derivative_s_coeff + derivative_s_exp
        derivative = derivative + derivative_p_coeff + derivative_p_exp

        return(np.sum(derivative, axis=1))

    def create_cofactor_matrix(self):
        pass

# Create both a valence and fitting object for the element
be_valence = GaussianValenceBasisSet(element, column_grid_points, file_path)
fitting_object_valence = Fitting(be_valence)

#fitting_object_valence.forward_greedy_algorithm(2.0, 0.01, np.copy(be_valence.electron_density_core), maximum_num_of_functions=100)