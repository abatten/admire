import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import h5py

import cmasher
from glob import glob

from pyx import plot_tools, param_tools
from pyx import print_tools
from pyx import math_tools

from model import DMzModel

import cmasher

#the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc('ytick', labelsize=16)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=18)
plt.rc('axes', labelsize=18)
plt.rc("xtick", top=True)
plt.rc("ytick", right=True)

def calc_std_bin_idx_from_pdf(pdf, num_sigma=1):
    """
    Calculates the bin index of a standard deviation from a pdf.

    """

    std_prop = {
        1: 0.682689492,
        2: 0.954499736,
        3: 0.997300204,
        4: 0.99993666,
        5: 0.999999426697,
    }

    std_limits = {
        1: ((1 - std_prop[1]) / 2, (1 + std_prop[1]) / 2),
        2: ((1 - std_prop[2]) / 2, (1 + std_prop[2]) / 2),
        3: ((1 - std_prop[3]) / 2, (1 + std_prop[3]) / 2),
        4: ((1 - std_prop[4]) / 2, (1 + std_prop[4]) / 2),
        5: ((1 - std_prop[5]) / 2, (1 + std_prop[5]) / 2),
    }


    std_lower_thresh, std_upper_thresh = std_limits[num_sigma]

    cdf = np.cumsum(pdf)
    cdf = cdf/cdf[-1]


    std_lower_idx = np.where(cdf >= std_lower_thresh)[0][0]
    std_upper_idx = np.where(cdf <= std_upper_thresh)[0][-1] + 1

    return std_lower_idx, std_upper_idx


def calc_std_values(bin_values, lower_idx, upper_idx, log=False):
    """

    Parameters
    ----------
    lower_idx: The index of the bin that corresponds to the lower limit uncertainty

    upper_idx: The index of the bin that corresponds to the upper limit uncertainty

    """
    if log:
        std_lower = 10**bin_values[lower_idx]
        std_upper = 10**bin_values[upper_idx]
    else:
        std_lower = bin_values[lower_idx]
        std_upper = bin_values[upper_idx]


    return std_lower, std_upper



def calc_sigma(pdf, bins, num_sigma=1):

    lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf, num_sigma=num_sigma)
    lower_std, upper_std = calc_std_values(bins, lower_idx, upper_idx)
    conf_interval_width = upper_std - lower_std
    return 0.5 * conf_interval_width



def plot_one_sigma(redshifts, sigma):

    #colours = np.array(list(map(mpl.colors.to_hex, cmasher.chroma(np.linspace(0.10, 0.90, 7)))))[[0, 2, 4, 3, 5, 1, 6]]
    #colours = ['red', 'blue', 'green', 'orange', 'black']

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    plt.plot(redshifts, sigma)


    plt.xlabel('Redshift')
    plt.ylabel('DM Cumulative Sum Sigma RefL0050N0752\n [$\mathrm{pc\ cm^{-3}}$]')
    plt.legend(frameon=False, fontsize=16)
    plt.savefig("TESTING_REFL0025N0376_CONF_VAR.png", dpi=200)


def run(params):

    # Create the file paths for the master file and the output file.
    masterfile = os.path.join(params["datadir"], params["masterfilename"] + ".hdf5")
    outputfile = os.path.join(params["outputdir"], params["outputfilename"])

    # Open the master and output file
    with h5py.File(masterfile, mode="r") as master, open(outputfile, mode='w') as output:
        num_slices = params["num_slices"]
        redshifts = np.zeros(num_slices)
        conf_interval_variance = np.zeros(num_slices)
        means = np.zeros(num_slices)
        conf_interval_width = np.zeros(num_slices)

        hist_list = []

        pbar = tqdm(range(num_slices))
        for index in pbar:
            pbar.set_description(f"Slice: {index}")

            # Load DM map data
            data = master["dm_maps"][f"slice_{index:03d}"]["DM"]

            # Correct the redshift to be the right hand side of the bin.
            redshifts[index] = master["dm_maps"][f"slice_{index:03d}"]["HEADER"].attrs["Redshift"] + math_tools.cosmology.cMpc_to_z(25)

            means[index] = np.mean(data)


            # Calculate the histogram
            bins = 10**np.linspace(0, 5, 1001)
            hist, edges = np.histogram(data, bins=bins, density=True)
            bin_centres = math_tools.calc_bin_centre(edges)

            # Normalise histogram to  pdf
            Bin_Widths = np.diff(edges)
            pdf = hist/Bin_Widths/np.sum(hist)

            conf_interval_width[index] = calc_sigma(pdf, bins)
            conf_interval_variance[index] = conf_interval_width[index]**2

        var_cumsum = np.cumsum(conf_interval_variance)
        std_cumsum = np.sqrt(var_cumsum)

        print(redshifts)
        print(conf_interval_variance)
        print(means)
        print(conf_interval_width)
        output.write("{:<12s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s}".format(
            "Redshift", "Variance", "Mean", "Std", "Var_CumSum", "CumSum_Std" "\n"))

        for z, var, mu, std, varcs, svarcs in zip(redshifts, conf_interval_variance, means, conf_interval_width, var_cumsum, std_cumsum):
            output.write(f"{z:<12.4f} {var:<12.4f} {mu:<12.4f} {std:<12.4f} {varcs:<12.4f} {svarcs:<12.4f}\n")










if __name__ == "__main__":
    colours = np.array(
        list(
            map(mpl.colors.to_hex, cmasher.rainforest(np.linspace(0.15, 0.80, 4)))

            )
        )

    plot_style_dict = {
        "RefL0100N1504": ("#000000", 3, "-"),
        "RefL0025N0376": (colours[2], 2, ":"),
        "RefL0025N0752": (colours[1], 2, "--"),
        "RefL0050N0752": (colours[0], 2, ":"),
        "RecalL0025N0752": (colours[3], 2, "--"),

    }



    print_tools.script_info.print_header("ADMIRE PIPELINE")
    #params = param_tools.dictconfig.read(sys.argv[1], "VarianceCorrelation")
    #run(params)

    colours = np.array(list(map(mpl.colors.to_hex, cmasher.chroma(np.linspace(0.10, 0.90, 7)))))[[0, 2, 4, 3, 5, 1, 6]]
    simulations = ["RefL0025N0376", "RefL0025N0752", "RecalL0025N0752", "RefL0050N0752", "RefL0100N1504"]
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    for idx, sim in enumerate(simulations):
        data_file = f"/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_{sim}/all_snapshot_data/output/T4EOS/confidence_interval_varience.txt"
        print(data_file)
        data = np.loadtxt(data_file, unpack=True, skiprows=1)
        z, var, mean, std, var_cumsum, cumsum_std = data


        plt.plot(z, cumsum_std, color=plot_style_dict[sim][0], label=sim,
                linewidth=plot_style_dict[sim][1], linestyle=plot_style_dict[sim][2])


    plt.xlabel('Redshift')
    plt.ylabel('$\mathrm{Cumulative}\ \sigma_\mathrm{CI}$\n [$\mathrm{pc\ cm^{-3}}$]')
    plt.legend(frameon=False, fontsize=16)
    plt.savefig("CONFIDENCE_INTERVAL_VARIANCE_2.pdf", dpi=200)




    #plot_one_sigma(z, cumsum_std)
    #plot_one_sigma(z, cumsum_std/np.cumsum(mean))
    #plot_one_sigma(z, np.sqrt(np.cumsum((std/mean)**2.0)))







    #c
    #colours = ['red', 'blue', 'green', 'orange', 'black']













    print_tools.script_info.print_footer()



