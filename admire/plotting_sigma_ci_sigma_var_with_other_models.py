import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy import cosmology
import astropy.units as apu

from scipy.integrate import quad

import cmasher


from pyx import math_tools
def set_rc_params(usetex=False):
    """
    Set the rcParams that will be used in all the plots.
    """

    rc_params = {
#        "axes.prop_cycle": cycler('color',
#            ['#1b9e77','#d95f02','#7570b3',
#             '#e7298a','#66a61e','#e6ab02',
#             '#a6761d','#666666']),
        "axes.labelsize": 18,
        "figure.dpi": 100,
        "legend.fontsize": 15,
        "legend.frameon": False,
        "text.usetex": usetex,
        "xtick.direction": 'in',
        "xtick.labelsize": 14,
        "xtick.minor.visible": True,
        "xtick.top": True,
        "ytick.direction": 'in',
        "ytick.labelsize": 14,
        "ytick.minor.visible": True,
        "ytick.right": True,
    }

    return rc_params

def make_lookback_axis(ax, cosmo, max_redshift):
    ax2 = ax.twiny()

    minor_tick_labels = np.arange(0, 13, 2)
    major_tick_labels = np.arange(1, 13, 2)

    major_tick_loc = np.zeros(len(major_tick_labels))
    minor_tick_loc = np.zeros(len(minor_tick_labels))

    for idx, labels in enumerate(zip(major_tick_labels, minor_tick_labels)):
        major, minor = labels

        if major < 0.0001:
            major_tick_loc[idx] = 0
        else:
            major_tick_loc[idx] = cosmology.z_at_value(cosmo.lookback_time, apu.Gyr * major) / max_redshift

        if minor < 0.0001:
            minor_tick_loc[idx] = 0
        else:
            minor_tick_loc[idx] = cosmology.z_at_value(cosmo.lookback_time, apu.Gyr * minor) / max_redshift



    ax2.set_xticks(major_tick_loc)
    ax2.set_xticks(minor_tick_loc, minor=True)

    major_tick_labels = [f"$\mathrm{x:.0f}$" for x in major_tick_labels]
    ax2.set_xticklabels(major_tick_labels, fontsize=14)

    return ax2


def plot_fit(plot_params, data_params=None, sigma_models=None, stat_types=["Mean"], simname="RefL0100N1504"):

    sim_idx_dict = {
        "RefL0100N1504": 0,
        "RefL0025N0376": 1,
        "RefL0025N0752": 2,
        "RecalL0025N0752": 3,
        "RefL0050N0752": 4,
    }
    sim_idx = sim_idx_dict[simname]


    stat_type_dict = {
        "Mean": 1,
        "Median": 2,
        "sigmaCI": 8,
        "sigmaVar": 4,
    }


    # Set up figure
    plt.rcParams.update(set_rc_params(usetex=True))
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    sim_data = np.loadtxt(data_params["filename"], unpack=True, skiprows=2)

    redshifts = sim_data[0]

    for stat_name in stat_types:
        stat_idx = stat_type_dict[stat_name]
        stat = sim_data[stat_idx]


    # Plot SigmaCI and SigmaVar
        ax.plot(redshifts, stat,
            color=plot_params[f"{stat_name}_data_lcolor"],
            linewidth=plot_params[f"{stat_name}_data_lwidth"],
            linestyle=plot_params[f"{stat_name}_data_lstyle"],
            label=plot_params[f"{stat_name}_data_label"])

    output_name = \
        f"./analysis_plots/shuffled/{plot_params['output_name']}{plot_params['output_format']}"


    for model in sigma_models.keys():

        z, sigma = np.loadtxt(sigma_models[model]["data_loc"], unpack=True, delimiter=" ")

        plt.plot(z, sigma, color=sigma_models[model]["color"], label=sigma_models[model]["label"])





    print(output_name)

    ax.set_ylim(0.000001, 300)
    ax.set_xlim(0.000001, 3.013)
    p13 = cosmology.Planck13
    ax_twin = math_tools.cosmology.make_lookback_time_axis(ax, cosmo=p13, z_range=(redshifts[0], redshifts[-1]))
    ax_twin.set_xlabel("$\mathrm{Lookback\ Time\ [Gyr]}$")

    ax.set_ylabel(plot_params['ylabel'])
    ax.set_xlabel("Redshift")

    ax.legend()
    plt.savefig(output_name)


if __name__ == "__main__":
    sigma_plot_params = {
        "output_name": "Sigma_with_other_models",
        "output_format": ".png",
        "ylabel": "$\sigma \ \mathrm{[pc\ cm^{-3}]}$",
        "sigmaCI_data_lstyle": "-",
        "sigmaCI_data_lwidth": 4,
        "sigmaCI_data_lcolor": "#000000",
        "sigmaCI_data_label": "$\mathrm{This\ Work}\ \sigma_\mathrm{CI}$",
        "sigmaVar_data_lstyle": "--",
        "sigmaVar_data_lwidth": 4,
        "sigmaVar_data_lcolor": "#000000",
        "sigmaVar_data_label": "$\mathrm{This\ Work}\ \sigma_\mathrm{Var}$",
    }

    data_params = {
        "filename": "./analysis_outputs/shuffled/ANALYSIS_RefL0100N1504_mean_var_std_from_pdf.txt"
    }


    sigma_models = {
        # "dolag": {
        #     "data_loc": "./Sigma_redshift_models/dolag2015.txt",
        #     "label": "Dolag et al. 2015",
        #     "color": "green",
        # },
        "mcquinn": {
            "data_loc": "./Sigma_redshift_models/mcquinn2014.txt",
            "label": "Mcquinn 2014",
            "color": "green",
        },
        "jaro": {
            "data_loc": "./Sigma_redshift_models/jaroszynski2019.txt",
            "label": "Jaroszynski 2019",
            "color": "red",
        }

    }




    plot_fit(sigma_plot_params, data_params=data_params, sigma_models=sigma_models, stat_types=["sigmaCI", "sigmaVar"], simname="RefL0100N1504")
