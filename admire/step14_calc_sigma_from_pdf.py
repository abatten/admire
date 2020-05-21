import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmasher
from glob import glob

from pyx import plot_tools
from pyx import print_tools
from pyx import math_tools

from model import DMzModel

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



def calc_sigma(model):

    model_label = model.label
    output = open(f"one_sigma_EAGLE_from_PDF_{model_label}.txt", "w")
    HEADER = "{:<16s} {:<16s}\n".format("Redshift", model_label)
    output.write(HEADER)

    if model.category == '2D-hydrodynamic':
        redshift = model.z_bins + math_tools.cosmology.cMpc_to_z(model.boxsize)

        std1_values = np.zeros(len(model.z_bins))
        #std1_values = np.zeros(len(model.z_bins))
        #std1_values = np.zeros(len(model.z_bins))
        for z_idx, pdf in enumerate(model.Hist.T):
            lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf)
            lower_std, upper_std = calc_std_values(model.DM_bins, lower_idx, upper_idx)
            std1_values[z_idx] = upper_std - lower_std

    for z, std in zip(redshift, std1_values):
        output.write(f"{z:<16.4f} {std:<16.4f}\n")
            #    lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf, num_sigma=2)
            #    lower_std, upper_std = calc_std_values(model.DM_bins, lower_idx, upper_idx)
            #    std2_values.append((lower_std, upper_std))
                
            #    lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf, num_sigma=3)
            #    lower_std, upper_std = calc_std_values(model.DM_bins, lower_idx, upper_idx)
            #    std3_values.append((lower_std, upper_std))
    
    output.close()
 


def plot_one_sigma(models):

    redshift_list = []
    std_list = []
    label_list = []

    colours = np.array(list(map(mpl.colors.to_hex, cmasher.chroma(np.linspace(0.10, 0.90, 7)))))[[0, 2, 4, 3, 5, 1, 6]]
    #colours = ['red', 'blue', 'green', 'orange', 'black']

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    for idx, model in enumerate(models):
        z, std = np.loadtxt(f"one_sigma_EAGLE_from_PDF_{model.label}.txt", unpack=True, skiprows=1)

        plt.plot(z, std, label=model.label, color=colours[idx])
        redshift_list.append(z)
        std_list.append(std)
        label_list.append(model.label)

    
    plt.xlabel('Redshift')
    plt.ylabel('DM 68\% confidence interval width \n [$\mathrm{pc cm^{-3}}$]')
    plt.legend(frameon=False, fontsize=16)
    plt.savefig("TESTING_STD_FROM_PDF.png", dpi=200)




if __name__ == "__main__":
    
    RefL0025N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0752/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "label"        : "RefL0025N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : 'orange',
        "linestyle"    : '-',
        "linewidth"    : 2,
        "marker"       : None,
        "plot_toggle"  : True,
        "boxsize"      : 25
    }
    
    RefL0025N0376 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "label"        : "RefL0025N0376",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : 'blue',
        "linestyle"    : ':',
        "linewidth"    : 2,
        "marker"       : None,
        "plot_toggle"  : True,
        "boxsize"      : 25
    }
    
    RecalL0025N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RecalL0025N0752/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "label"        : "RecalL0025N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : 'green',
        "linestyle"    : ':',
        "linewidth"    : 2,
        "marker"       : None,
        "plot_toggle"  : True,
        "boxsize"      : 25
    }
    
    RefL0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0050N0752/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed_bin_width_and_idx_corrected.hdf5",
        "label"        : "RefL0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : 'green',
        "linestyle"    : ':',
        "linewidth"    : 2,
        "marker"       : None,
        "plot_toggle"  : True,
        "boxsize"      : 50
    }
    
    RefL0100N1504 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_normed_bin_width_and_idx_corrected.hdf5",      
        "label"        : "RefL0100N1504",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : 'deepskyblue',
        "linestyle"    : ':',
        "linewidth"    : 2,
        "marker"       : None,
        "plot_toggle"  : True,
        "boxsize"      : 100
    }
    
    model_dicts = [
        RefL0025N0752,
        RefL0025N0376,
        RecalL0025N0752,
        RefL0050N0752,
        RefL0100N1504
    ]

#######################################################################
#######################################################################
#                         DO NOT TOUCH
#######################################################################
#######################################################################

    for model in model_dicts:
        path = os.path.join(model["dir_name"], model["file_name"])
        model["path"] = path

    all_models = []
    for model_dict in model_dicts:
        model = DMzModel(model_dict)
        all_models.append(model)

    for model in all_models:
        calc_sigma(model)

    plot_one_sigma(all_models)

    print_tools.print_footer()
