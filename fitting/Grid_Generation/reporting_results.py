import os
from fitting.grid_generation import Grid_1D_Generator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    directory =os.getcwd()
    results_dir = directory + "\Results"
    print(results_dir)


    effort = 10
    type_of_grid = "CC"
    delayed_seq = np.arange(1, effort + 1)
    all_grids_obj = Grid_1D_Generator(type_of_grid, effort, delayed_seq)


    list_of_colours = ["b", "g", "r", "c", "m", "y", "k", "w"]
    plt.title("Clenshaw-Curtis 1D Grid Points Between 0 and 1")
    for x in range(0, effort):
        plt.plot(all_grids_obj.all_1D_grid_objs[x].grid_1D, list_of_colours[x % 8] + "o", label="effort="+str(x + 1)+", num_pts=" +
                 str(all_grids_obj.all_1D_grid_objs[x].num_of_points))
    plt.ylabel("Value of Grid Point")
    plt.xlabel("Grid Point Index")
    plt.legend()
    plt.figtext(.12, .02, "delayed sequence is [1, 2, 3, ..., 9, 10 = (effort)]")
    plt.savefig(results_dir + r'\2016-06-06_CC_Grid_Points.png')
    plt.show()


    plt.title("Clenshaw-Curtis 1D Weights")
    for x in range(0, effort - 5):
        plt.plot(all_grids_obj.all_1D_grid_objs[x].weights_1D, list_of_colours[x % 8] + "o", label="effort="+str(x + 1)+", num_pts=" +
                 str(all_grids_obj.all_1D_grid_objs[x].num_of_points))
    plt.legend()
    plt.figtext(.12, .02, "delayed sequence is [1, 2, 3, ..., 9, 10 = (effort)]")
    plt.savefig(results_dir + r'\2016-06-06_CC_Weights_Points_1.png')
    plt.show()

    plt.title("Clenshaw-Curtis 1D Weights")
    for x in range(6, effort ):
        plt.plot(all_grids_obj.all_1D_grid_objs[x].weights_1D, list_of_colours[x % 8] + "o", label="effort="+str(x + 1)+", num_pts=" +
                 str(all_grids_obj.all_1D_grid_objs[x].num_of_points))
    plt.legend()
    plt.figtext(.12, .02, "delayed sequence is [1, 2, 3, ..., 9, 10 = (effort)]")

    plt.savefig(results_dir + r'\2016-06-06_CC_Weights_Points_2.png')
    plt.show()