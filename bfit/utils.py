# -*- coding: utf-8 -*-
# FittingBasisSets is a basis-set curve-fitting optimization package.
#
# Copyright (C) 2018 The FittingBasisSets Development Team.
#
# This file is part of FittingBasisSets.
#
# FittingBasisSets is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# FittingBasisSets is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
r"""Contain plotting utility functions for fit_densities file."""

import matplotlib.pyplot as plt
import os

__all__ = ["plot_error", "plot_model_densities", "create_plots"]


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
        for model in additional_models_plots:
            ax.semilogy(grid_pts, model_dens, 'o:', lw=3, label="Gaussian Fitted Electron Density",
                        color=tableau20[7])
    # plt.xlim(0, 25.0*0.5291772082999999)
    plt.xlim(0, 9)
    plt.ylim(ymin=1e-9)
    ax.set_axis_bgcolor('white')
    plt.xlabel('Distance from the nucleus [A]')
    plt.ylabel(r'$Log(\rho(r_{Bohr}^{-3}))$', fontsize=16)
    plt.title(title, fontweight='bold')
    plt.grid(color=tableau20[-2])
    plt.legend()
    directory = os.path.dirname(__file__).rsplit('/', 2)[0] + "/results/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/" + figure_name + ".png")
    plt.close()


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
    axarr[0, 0].plot([1] + [x + x - 1 for x in range(2, len(errors[2]) + 1)], errors[0], 'o-',
                     color=tableau20[0])
    axarr[0, 0].set_title('Integrated Fitted Density Model')
    axarr[0, 0].set_ylabel(r'$\int \rho^o(r) 4 \pi r^2 dr$')
    axarr[0, 1].semilogy([1] + [x + x - 1 for x in range(2, len(errors[2]) + 1)], errors[1], 'o-',
                         color=tableau20[0])
    axarr[0, 1].set_title('Absolute Difference In Models')
    axarr[0, 1].set_ylabel(r'$\int |\rho(r) - \rho^o(r)| dr$')
    axarr[1, 0].semilogy([1] + [x + x - 1 for x in range(2, len(errors[2]) + 1)], errors[2], 'o-',
                         color=tableau20[0])
    axarr[1, 0].set_title("Absolute Difference Times Radius Squared")
    axarr[1, 0].set_xlabel("Number of Functions")
    axarr[1, 0].set_ylabel(r'$\int |\rho(r) - \rho^o(r)| r^2 dr$')
    axarr[1, 1].semilogy([1] + [x + x - 1 for x in range(2, len(errors[2]) + 1)], errors[3], 'o-',
                         color=tableau20[0])
    axarr[1, 1].set_title('Kullback-Leiger Function Value')
    axarr[1, 1].set_ylabel(r'$\int \rho(r) \frac{\rho(r)}{\rho^o(r)} 4 \pi r^2 dr$')
    axarr[1, 1].set_xlabel("Number of Functions")
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    directory = os.path.dirname(__file__).rsplit('/', 2)[0] + "/bfit/results/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/" + figure_name + ".png")
    plt.close()


def create_plots(self, integrate, error1, error2, obj_func):
    directory = os.path.dirname(__file__).rsplit('/', 2)[0] + "/bfit/results/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.plot(integrate)
    plt.title("Integration of Model Using Trapz")
    plt.xlabel("Num of Iterations")
    plt.ylabel("Integration of Model Using Trapz")
    plt.savefig(directory + self.element_name + "_Integration_Trapz.png")
    plt.close()

    plt.semilogy(error1)
    plt.xlabel("Num of Iterations")
    plt.title("Goodness of Fit")
    plt.ylabel("Int |Model - True| dr")
    plt.savefig(directory + self.element_name + "_good_of_fit.png")
    plt.close()

    plt.semilogy(error2)
    plt.xlabel("Num of Iterations")
    plt.title("Goodness of Fit with r^2")
    plt.ylabel("Int |Model - True| r^2 dr")
    plt.savefig(directory + self.element_name + "_goodness_of_fit_r_squared.png")
    plt.close()

    plt.semilogy(obj_func)
    plt.xlabel("Num of Iterations")
    plt.title("Objective Function")
    plt.ylabel("KL Divergence Formula")
    plt.savefig(self.element_name + "_objective_function.png")
    plt.close()
