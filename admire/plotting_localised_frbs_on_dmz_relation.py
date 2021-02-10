import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from astropy import cosmology

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


def calc_mean_from_pdf(bin_values, pdf):
    """
    """
    return  np.sum(bin_values[1:] * pdf) / np.sum(pdf)






def plot_sigmaz_relation(models, plot_output_name="sigmaz_relation",
                      plot_output_path=None, plot_output_format=".eps",
                      relative=False, z_min=0.0, z_max=0.7, sigma_min=0, sigma_max=4000,
                      axis=None, verbose=True, frb_obs=None):
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
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)


    # Create a 2D image with different x-y bin sizes.
    #im = plot_tools.plot_2d_array(
    #    np.log10(models[0].Hist),
    #    xvals=models[0].z_bins,
    #    yvals=models[0].DM_bins,
    #    cmap=cmasher.arctic_r,
    #    passed_ax=ax,
    #    label="This Work",
    #    vmax=-2,
    #    vmin=-5
    #    )

    # Set up the colour bar
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(im, cax=cax, label=r"$\mathrm{PDF}$")
    #cbar.ax.tick_params(axis='y', direction='out')


    # Plot the 2D histogram as an image in the backgound
    for model in models:
        if model.category == '2D-hydrodynamic' and model.plot_toggle:
            print_tools.vprint(f"{model.label}", verbose=verbose)

            #filename = "RefL0100N1504_log_normal_fit_mean_std_percent.txt"
            #redshift, mean, std, percent = np.loadtxt(filename, unpack=True, skiprows=1)
            #output = open("RefL0100N1504_mean_direct_values.txt", "w")
            #output.write("Redshift Mean\n")
            #output.write("0.000 0.000\n")
            #redshift = model.z_bins + math_tools.cosmology.cMpc_to_z(100)

            #mean = np.zeros(len(model.z_bins))
            #for idx, pdf in enumerate(model.Hist.T):
            #    mean[idx] = calc_mean_from_pdf(model.DM_bins, pdf)
            #    z = redshift[idx]

            #    output.write(f"{redshift[idx]:.3f} {mean[idx]:.3f}\n")

            #output.close()

            data = np.loadtxt("analysis_outputs/shuffled/ANALYSIS_RefL0100N1504_confidence_intervals.txt", unpack=True, skiprows=2)
            redshift = data[0]
            mean = data[1]
            median = data[2]

            std1_values = []
            std2_values = []
            std3_values = []
            for idx, z in enumerate(redshift):
                std1_values.append((data[3][idx], data[4][idx]))
                std2_values.append((data[5][idx], data[6][idx]))
                std3_values.append((data[7][idx], data[8][idx]))


            mean_line = ax.plot(redshift, mean, color='black', label="$\mathrm{RefL0100N1504}\ <\mathrm{DM}>$", linewidth=2, zorder=1)

            #std1_values = []
            #std2_values = []
            #std3_values = []
            #for z_idx, pdf in enumerate(model.Hist.T):
                #lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf)
                #lower_std, upper_std = calc_std_values(model.DM_bins, lower_idx, upper_idx)
                #std1_values.append((lower_std, upper_std))

                #lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf, num_sigma=2)
                #lower_std, upper_std = calc_std_values(model.DM_bins, lower_idx, upper_idx)
                #std2_values.append((lower_std, upper_std))

                #lower_idx, upper_idx = calc_std_bin_idx_from_pdf(pdf, num_sigma=3)
                #lower_std, upper_std = calc_std_values(model.DM_bins, lower_idx, upper_idx)
                #std3_values.append((lower_std, upper_std))

            if frb_obs is not None:
                frb_handles = []
                frb_labels = []


                frb_repeater_list = []
                frb_non_repeater_list = []
                for frb in frb_obs_list:
                    if frb["Repeater"]:
                        frb_repeater_list.append(frb)
                    else:
                        frb_non_repeater_list.append(frb)

                colours = cmasher.take_cmap_colors('cmr.sunburst', 2, cmap_range=(0.15, 0.75), return_fmt='hex')

                for idx, frb in enumerate(frb_repeater_list):
                    redshift = frb["z"]
                    dm = frb["DM"] - frb["DM_MW"]

                    lower_errorbar = np.array([130])
                    upper_errorbar = frb["DM_MW"]
                    errors = (lower_errorbar, upper_errorbar)


                    marker = "d"
                    colour = colours[0]

                    im = ax.errorbar(redshift, dm, yerr=errors, color=colour, marker=marker, markersize=5, zorder=10000)


                frb_handles.append(im)
                frb_labels.append("Repeater")


                for idx, frb in enumerate(frb_non_repeater_list):
                    redshift = frb["z"]
                    dm = frb["DM"] - frb["DM_MW"]

                    lower_errorbar = np.array([130])
                    upper_errorbar = frb["DM_MW"]
                    errors = (lower_errorbar, upper_errorbar)


                    marker = "o"
                    colour = colours[1]

                    im = ax.errorbar(redshift, dm, yerr=errors, color=colour, marker=marker, markersize=5, zorder=10000)

                frb_handles.append(im)
                frb_labels.append("Non-Repeater")



            std1_values = np.array(std1_values)
            std2_values = np.array(std2_values)
            std3_values = np.array(std3_values)

            sigma_colours = np.array(
                list(map(mpl.colors.to_hex,
                         cmasher.ocean(np.linspace(0.20, 0.50, 3)))
                )
            )

            sig3 = ax.fill_between(model.z_bins, std3_values.T[0], std3_values.T[1],
                            alpha=0.35, color=sigma_colours[2], label=f"{model.label} $3 \sigma$")
            sig2 = ax.fill_between(model.z_bins, std2_values.T[0], std2_values.T[1],
                            alpha=0.35, color=sigma_colours[1], label=f"{model.label} $2 \sigma$")
            sig1 = ax.fill_between(model.z_bins, std1_values.T[0], std1_values.T[1],
                            alpha=0.35, color=sigma_colours[0], label=f"{model.label} $1 \sigma$")




                    #ax.scatter(redshift, dm, color=frb["Color"], marker='o', label=frb["Label"], zorder=10000)






    ax.set_xlim(z_min, z_max)
    ax.set_ylim(sigma_min, sigma_max)

    p13 = cosmology.Planck13
    #ax.set_xlim(0, 0.7)
    ax.set_ylim(0.0001, 1100)
    ax1_twin = math_tools.cosmology.make_lookback_time_axis(ax, cosmo=p13, z_range=(z_min, z_max))
    ax1_twin.set_xlabel("$\mathrm{Lookback\ Time\ [Gyr]}$")



    # This forces the x-tick labels to be integers for consistancy.
    # This fixes a problem I was having where it changed from ints to floats
    # seemingly randomly for different number of models.
    #ax.xaxis.set_major_locator(MaxNLocator(integer=False))

    ax.set_xlabel(r"$\rm{Redshift}$")
    #ax.set_ylabel(r"$\rm{DM_{Obs} - DM_{MW,\ NE2001}\ \left[pc\ cm^{-3}\right] }$")
    ax.set_ylabel(r"$\rm{DM_{cosmic}\ \left[pc\ cm^{-3}\right] }$")

    sigma_colours = np.array(
        list(map(mpl.colors.to_hex,
                    cmasher.ocean(np.linspace(0.20, 0.50, 3)))
        )
    )



    mean_line = mlines.Line2D([], [], color='black', linewidth=2, label=r"$\langle \mathrm{DM_{cosmic}} \rangle$")
    sig1_legend = mpatches.Patch(color=sigma_colours[0], label=r"$1\ \sigma_\mathrm{CI}$", alpha=0.35)
    sig2_legend = mpatches.Patch(color=sigma_colours[1], label=r"$2\ \sigma_\mathrm{CI}$", alpha=0.35)
    sig3_legend = mpatches.Patch(color=sigma_colours[2], label=r"$3\ \sigma_\mathrm{CI}$", alpha=0.35)


    # Create legends for the FRBs and EAGLE data
    # They need to be seperate to make better use of space.
    sigma_legend = ax.legend(handles=[mean_line, sig1_legend, sig2_legend, sig3_legend], loc='upper left', frameon=False, fontsize=11)
    frb_legend = ax.legend(handles=frb_handles, labels=frb_labels, loc='lower right', fontsize=11, frameon=False)

    # Add the legends manually to the current Axes.
    ax.add_artist(sigma_legend)
    ax.add_artist(frb_legend)


    output_file_name = os.path.join(plot_output_path, f"{plot_output_name}{plot_output_format}")
    plt.savefig(output_file_name, dpi=300)




def calc_least_squares_frb(frb_obs):
    data = np.loadtxt("analysis_outputs/shuffled/ANALYSIS_RefL0100N1504_confidence_intervals.txt", unpack=True, skiprows=2)
    redshift = data[0]
    mean = data[1]
    median = data[2]

    sigma_ci = []

    std1_values = []
    for idx, z in enumerate(redshift):
        sigma_ci.append((data[4][idx] - data[3][idx]) / 2)




    from scipy.interpolate import interp1d

    model = interp1d(mean, redshift)
    model_ci = interp1d(redshift, sigma_ci)
    least_sq = 0
    for frb in frb_obs:
        estimate = model(frb["DM"] - frb["DM_MW"])
        estimate_ci = model_ci(estimate)

        measured = frb["z"]

        frb_var = ((frb["DM_MW"] + 130) / 2)**2

        var = estimate_ci**2 #+ ((frb["DM_MW"] + 130) / 2)**2


        print(estimate, measured)

        least_sq += (measured - estimate)**2 / estimate

    print(least_sq)



if __name__ == "__main__":
    print_tools.print_header("Sigma-z Relation")

    output_file_name = "analysis_plots/shuffled/RefL0100N1504_three_sigma_CI_with_localised_frbs_NEW"


    ##############################################################
    # LOCALISED FRB'S DATA

    frb_obs_list = [

        # CHIME Localised FRB
        {"DM": np.array([348.79]),
        "DM_MW": np.array([199]),
        "z": np.array([0.0337]),
        "z_err": np.array([0.0002]),
        "Label": "FRB 180916",# (Marcote+2020)",
        "Color": "#4d0378",
        "Marker": "D",
        "Repeater": True,
        },

        # Macquart 2020
        {"DM": np.array([339.5]),
        "DM_MW": np.array([37.2]),
        "z": np.array([0.1178]),
        "z_err": np.array([0]),
        "Label": "FRB 190608",# (Macquart+2020)",
        "Color": "black",
        "Marker": "o",
        "Repeater": False,
        },

        # Macquart 2020
        {"DM": np.array([380]),
        "DM_MW": np.array([27.2]),
        "z": np.array([0.160]),
        "z_err": np.array([0]),
        "Label": "FRB 2004330",# (Macquart+2020)",
        "Color": "black",
        "Marker": "o",
        "Repeater": False,
        },


        #Original Repeater
        {"DM": np.array([557]),
        "DM_MW": np.array([188]),
        "z": np.array([0.19273]),
        "z_err": np.array([0.000]),
        "Label": "FRB 121102",# (Tendulkar+2017)",
        "Color": "Grey",
        "Marker": "D",
        "Repeater": True,
        },

        {"DM": np.array([507.9]),
        "DM_MW": np.array([44.2]),
        "z": np.array([0.2340]),
        "z_err": np.array([0.000]),
        "Label": "FRB 191001",# (Bhundari+2020)",
        "Color": "Grey",
        "Marker": "D",
        "Repeater": False,
        },

        {"DM": np.array([504.1]),
        "DM_MW": np.array([38.5]),
        "z": np.array([0.2365]),
        "z_err": np.array([0.000]),
        "Label": "FRB 191001",# (Heinz+2020)",
        "Color": "Grey",
        "Marker": "D",
        "Repeater": False,
        },


        # Macquart 2020
        {"DM": np.array([364.5]),
        "DM_MW": np.array([57.3]),
        "z": np.array([0.2913]),
        "z_err": np.array([0]),
        "Label": "FRB 190102",# (Macquart+2020)",
        "Color": "black",
        "Marker": "o",
        "Repeater": False,
        },

        # Bannister Localised FRB
        {"DM": np.array([361.42]),
        "DM_MW": np.array([40.5]),
        "z": np.array([0.3214]),
        "z_err": 0,
        "Label": "FRB 180924",# (Bannister+2019)",
        "Color": '#900c5c',
        "Marker": "o",
        "Repeater": False,
         },

        # Macquart 2020
        {"DM": np.array([321.4]),
        "DM_MW": np.array([57.8]),
        "z": np.array([0.378]),
        "z_err": np.array([0.000]),
        "Label": "FRB 190611",# (Macquart+2020)",
        "Color": "black",
        "Marker": "o",
        "Repeater": False,
        },

        #Prochaska Localised FRB
        {"DM": np.array([589.27]),
        "DM_MW": np.array([40.2]),
        "z": np.array([0.47550]),
        "z_err": np.array([0.000]),
        "Label": "FRB 181112",# (Prochaska+2019)",
        "Color": "#b84721",
        "Marker": "o",
        "Repeater": False,
        },

        # Macquart 2020
        {"DM": np.array([593.1]),
        "DM_MW": np.array([56.5]),
        "z": np.array([0.522]),
        "z_err": np.array([0.000]),
        "Label": "FRB 190711",# (Macquart+2020)",
        "Color": "Black",
        "Marker": "o",
        "Repeater": True,
        },

        # Ravi Localised FRB
        {"DM": np.array([760.8]),
        "DM_MW": np.array([37]),
        "z": np.array([0.66]),
        "z_err": 0,
        "Label": "FRB 190523",# (Ravi+2019)",
        "Color": '#b18d05',
        "Marker": "o",
        "Repeater": False,
        },











    ]
##############################################################

    RefL0100N1504 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_bin_width_and_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RefL0100N1504",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : '#FF6800',
        "linestyle"    : ':',
        "linewidth"    : 2,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    model_dicts = [
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


    plot_sigmaz_relation(all_models, output_file_name, "", frb_obs=frb_obs_list, plot_output_format=".png")

    calc_least_squares_frb(frb_obs_list)

    print_tools.print_footer()
