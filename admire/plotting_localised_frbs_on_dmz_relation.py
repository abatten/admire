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

from model import DMzModel

#the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=20)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=20)
plt.rc('axes', labelsize=20)


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

def plot_sigmaz_relation(models, plot_output_name="sigmaz_relation", 
                      plot_output_path=None, plot_output_format=".png", 
                      relative=False, z_min=0.0, z_max=3.0, sigma_min=0, sigma_max=4000,
                      axis=None, verbose=True):
    """
    Plots the Sigma-z Relation


    Parameters
    ----------
    models: 

    plot_output_name: str, optional
        The name of the output plot. Default: "dmz_relation".

    plot_output_path: str, optional

    plot_output_format: str, optional

    z_min: int or float, optional
        The minimum Default: 0.0

    z_max: int or float, optional
        The maximum redshift (max x limit) for the 

    """

    # Create an figure instance, but use a passed axis if given.
    if axis:
        ax = axis
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)


    # Create a 2D image with different x-y bin sizes.
    im = plot_tools.plot_2d_array(
        models[0].Hist, 
        xvals=models[0].z_bins,
        yvals=models[0].DM_bins,
        cmap=cmasher.rainforest_r,
        passed_ax=ax
        )

    # Set up the colour bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, label=r"$\mathrm{PDF}$")
    cbar.ax.tick_params(axis='y', direction='out')


    # Plot the 2D histogram as an image in the backgound
    for model in models:
        if model.category == '2D-hydrodynamic' and model.plot_toggle:
            print_tools.vprint(f"{model.label}", verbose=verbose)
           

            std_values = []
            for z_idx, pdf in enumerate(model.Hist.T):
                lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf)
                lower_std, upper_std = calc_std_values(model.DM_bins, lower_idx, upper_idx)
                std_values.append((lower_std, upper_std))
            
            std_values = np.array(std_values)
            ax.fill_between(model.z_bins, std_values.T[0], std_values.T[1],
                            alpha=0.5, color=model.color, label=model.label)




    ax.set_xlim(z_min, z_max)
    ax.set_ylim(sigma_min, sigma_max)

    # This forces the x-tick labels to be integers for consistancy.
    # This fixes a problem I was having where it changed from ints to floats 
    # seemingly randomly for different number of models. 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(r"$\rm{Redshift}$")
    ax.set_ylabel(r"$\rm{DM\ \left[pc\ cm^{-3}\right] }$")
    ax.legend(frameon=False, fontsize=10, loc="lower right" )


    plt.tight_layout()
    output_file_name = os.path.join(plot_output_path, f"{plot_output_name}{plot_output_format}")
    plt.savefig(output_file_name, dpi=200)










if __name__ == "__main__":
    print_tools.print_header("Sigma-z Relation")

    output_file_name = "sigmaz_relation_full_RefL0025N0376_RefL0025N0752_RecalL0025N0752_idx_corrected_background" 
    
    
    
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
    }
    
    model_dicts = [
        RefL0025N0752,
        RefL0025N0376,
        RecalL0025N0752
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
 

    plot_sigmaz_relation(all_models, output_file_name, "")

    print_tools.print_footer()


