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


def expon_relation(x, coeff, expon, inter):
    return coeff * np.exp(expon * x) + inter

def expon2_relation(x, coeff, expon):
    return coeff * (1 - np.exp(expon * x))


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









def plot_fit(plot_params, data_params=None, stat_types=["Mean"], simname="RefL0100N1504"):

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
    fig = plt.figure(figsize=(8,6))
    gs = fig.add_gridspec(10, 1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[:6, 0])
    ax2 = fig.add_subplot(gs[6:8, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[8:10, 0], sharex=ax1)



    sigmaCI_fit = np.loadtxt(plot_params["sigmaCI_data"], unpack=True, skiprows=1, usecols=range(1,9))
    sigmaVar_fit = np.loadtxt(plot_params["sigmaVar_data"], unpack=True, skiprows=1, usecols=range(1,9))
    sim_data = np.loadtxt(data_params["filename"], unpack=True, skiprows=2)

    redshifts = sim_data[0]


    sigmaCI_coeff = sigmaCI_fit[0][sim_idx]
    sigmaCI_expon = sigmaCI_fit[2][sim_idx]
    sigmaCI_inter = sigmaCI_fit[4][sim_idx]

    sigmaVar_coeff = sigmaVar_fit[0][sim_idx]
    sigmaVar_expon = sigmaVar_fit[2][sim_idx]
    sigmaVar_inter = sigmaVar_fit[4][sim_idx]

    sigmaCI_vals = expon_relation(redshifts, sigmaCI_coeff, sigmaCI_expon, sigmaCI_inter)
    sigmaVar_vals = expon_relation(redshifts, sigmaVar_coeff, sigmaVar_expon, sigmaVar_inter)

    stat_vals = {
        "sigmaCI": sigmaCI_vals,
        "sigmaVar": sigmaVar_vals,
    }

    plot_params["sigmaCI_fit_label"] += f"$= {sigmaCI_coeff:.1f}e^{{{sigmaCI_expon:.1f}z}} + {sigmaCI_inter:.1f}$"
    plot_params["sigmaVar_fit_label"] += f"$= {sigmaVar_coeff:.1f}e^{{{sigmaVar_expon:.1f}z}} + {sigmaVar_inter:.1f}$"

    ax2.plot(np.linspace(-1, 4, 100), np.zeros(100), "grey", linewidth=1)
    ax3.plot(np.linspace(-1, 4, 100), np.zeros(100), "grey", linewidth=1)

    for stat_name in stat_types:
        stat_idx = stat_type_dict[stat_name]
        stat = sim_data[stat_idx]
        stat_val = stat_vals[stat_name]

    # Plot SigmaCI and SigmaVar
        ax1.plot(redshifts, stat,
            color=plot_params[f"{stat_name}_data_lcolor"],
            linewidth=plot_params[f"{stat_name}_data_lwidth"],
            linestyle=plot_params[f"{stat_name}_data_lstyle"],
            label=plot_params[f"{stat_name}_data_label"])



        # Plot the SigmaCI Fit Rediduals
        ax2.plot(redshifts, stat - stat_val,
            color=plot_params[f"{stat_name}_fit_lcolor"],
            linewidth=plot_params[f"{stat_name}_fit_lwidth"],
            linestyle=plot_params[f"{stat_name}_fit_lstyle"],
            label=plot_params[f"{stat_name}_fit_label"])

        # Plot the Non-Linear Model Fit Relative Residuals
        ax3.plot(redshifts, (stat - stat_val)/stat * 100,
            color=plot_params[f"{stat_name}_fit_lcolor"],
            linewidth=plot_params[f"{stat_name}_fit_lwidth"],
            linestyle=plot_params[f"{stat_name}_fit_lstyle"],
            label=plot_params[f"{stat_name}_fit_label"])

        print(f"stat: {stat_name}",  "mean residual", np.mean(((stat - stat_val)/stat * 100)**2)**0.5)
    #print(f"stat: {stat_name}",  "mean nonlinear residual", np.mean(((stat - nonlin_dm_vals)/stat * 100)**2)**0.5 )

    # if stat_type == "Mean":
    #     ax2.axis(ymin=-45, ymax=45, xmin=0.0001, xmax=3.013)
    #     ax3.axis(ymin=-10, ymax=10, xmin=0.0001, xmax=3.013)
    # elif stat_type == "Median":
    #     ax2.axis(ymin=-45, ymax=45, xmin=0.0001, xmax=3.013)
    #     ax3.axis(ymin=-20, ymax=20, xmin=0.0001, xmax=3.013)
    for stat_name in stat_types:
        stat_idx = stat_type_dict[stat_name]
        stat = sim_data[stat_idx]
        stat_val = stat_vals[stat_name]
        # Plot the SigmaCI Fit
        ax1.plot(redshifts, stat_val,
            color=plot_params[f"{stat_name}_fit_lcolor"],
            linewidth=plot_params[f"{stat_name}_fit_lwidth"],
            linestyle=plot_params[f"{stat_name}_fit_lstyle"],
            label=plot_params[f"{stat_name}_fit_label"])




    output_name = \
        f"./analysis_plots/shuffled/{plot_params['output_name']}{plot_params['output_format']}"


    print(output_name)

    ax1.set_ylim(0.000001, 300)
    ax1.set_xlim(0.000001, 3.013)
    p13 = cosmology.Planck13
    ax1_twin = math_tools.cosmology.make_lookback_time_axis(ax1, cosmo=p13, z_range=(redshifts[0], redshifts[-1]))
    ax1_twin.set_xlabel("$\mathrm{Lookback\ Time\ [Gyr]}$")



    ax1.set_ylabel(plot_params['ylabel'])
    ax2.set_ylabel("\\textrm{Residual} \n \\textrm{(Absolute)}", fontsize=14)
    ax2.set_xlim(0.000001, 3.013)

    ax3.set_ylabel("\\textrm{Residual} \n \\textrm{(Relative \%)}", fontsize=14)
    ax3.set_xlabel("\\textrm{Redshift}")
    ax3.set_xlim(0.000001, 3.013)

    ax2.axis(ymin=-10.0, ymax=10.0, xmin=0.0001, xmax=3.013)
    ax3.axis(ymin=-4, ymax=4, xmin=0.0001, xmax=3.013)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.spines['bottom'].set_linewidth(0)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(output_name)




def plot_fit2(plot_params, data_params=None, stat_types=["Mean"], simname="RefL0100N1504"):

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
    fig = plt.figure(figsize=(8,6))
    gs = fig.add_gridspec(10, 1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[:6, 0])
    ax2 = fig.add_subplot(gs[6:8, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[8:10, 0], sharex=ax1)



    sigmaCI_fit = np.loadtxt(plot_params["sigmaCI_data"], unpack=True, skiprows=1, usecols=range(1,7))
    sigmaVar_fit = np.loadtxt(plot_params["sigmaVar_data"], unpack=True, skiprows=1, usecols=range(1,7))
    sim_data = np.loadtxt(data_params["filename"], unpack=True, skiprows=2)

    redshifts = sim_data[0]


    sigmaCI_coeff = sigmaCI_fit[0][sim_idx]
    sigmaCI_expon = sigmaCI_fit[2][sim_idx]


    sigmaVar_coeff = sigmaVar_fit[0][sim_idx]
    sigmaVar_expon = sigmaVar_fit[2][sim_idx]


    sigmaCI_vals = expon2_relation(redshifts, sigmaCI_coeff, sigmaCI_expon)
    sigmaVar_vals = expon2_relation(redshifts, sigmaVar_coeff, sigmaVar_expon)

    stat_vals = {
        "sigmaCI": sigmaCI_vals,
        "sigmaVar": sigmaVar_vals,
    }

    plot_params["sigmaCI_fit_label"] += f"$= {sigmaCI_coeff:.1f}(1 - e^{{{sigmaCI_expon:.1f}z}})$"
    plot_params["sigmaVar_fit_label"] += f"$= {sigmaVar_coeff:.1f}(1 - e^{{{sigmaVar_expon:.1f}z}})$"

    ax2.plot(np.linspace(-1, 4, 100), np.zeros(100), "grey", linewidth=1)
    ax3.plot(np.linspace(-1, 4, 100), np.zeros(100), "grey", linewidth=1)

    for stat_name in stat_types:
        stat_idx = stat_type_dict[stat_name]
        stat = sim_data[stat_idx]
        stat_val = stat_vals[stat_name]

    # Plot SigmaCI and SigmaVar
        ax1.plot(redshifts, stat,
            color=plot_params[f"{stat_name}_data_lcolor"],
            linewidth=plot_params[f"{stat_name}_data_lwidth"],
            linestyle=plot_params[f"{stat_name}_data_lstyle"],
            label=plot_params[f"{stat_name}_data_label"])



        # Plot the SigmaCI Fit Rediduals
        ax2.plot(redshifts, stat - stat_val,
            color=plot_params[f"{stat_name}_fit_lcolor"],
            linewidth=plot_params[f"{stat_name}_fit_lwidth"],
            linestyle=plot_params[f"{stat_name}_fit_lstyle"],
            label=plot_params[f"{stat_name}_fit_label"])

        # Plot the Non-Linear Model Fit Relative Residuals
        ax3.plot(redshifts, (stat - stat_val)/stat * 100,
            color=plot_params[f"{stat_name}_fit_lcolor"],
            linewidth=plot_params[f"{stat_name}_fit_lwidth"],
            linestyle=plot_params[f"{stat_name}_fit_lstyle"],
            label=plot_params[f"{stat_name}_fit_label"])

        print(f"stat: {stat_name}",  "mean residual", np.mean(((stat - stat_val)/stat * 100)**2)**0.5)
    #print(f"stat: {stat_name}",  "mean nonlinear residual", np.mean(((stat - nonlin_dm_vals)/stat * 100)**2)**0.5 )

    # if stat_type == "Mean":
    #     ax2.axis(ymin=-45, ymax=45, xmin=0.0001, xmax=3.013)
    #     ax3.axis(ymin=-10, ymax=10, xmin=0.0001, xmax=3.013)
    # elif stat_type == "Median":
    #     ax2.axis(ymin=-45, ymax=45, xmin=0.0001, xmax=3.013)
    #     ax3.axis(ymin=-20, ymax=20, xmin=0.0001, xmax=3.013)
    for stat_name in stat_types:
        stat_idx = stat_type_dict[stat_name]
        stat = sim_data[stat_idx]
        stat_val = stat_vals[stat_name]
        # Plot the SigmaCI Fit
        ax1.plot(redshifts, stat_val,
            color=plot_params[f"{stat_name}_fit_lcolor"],
            linewidth=plot_params[f"{stat_name}_fit_lwidth"],
            linestyle=plot_params[f"{stat_name}_fit_lstyle"],
            label=plot_params[f"{stat_name}_fit_label"])




    output_name = \
        f"./analysis_plots/shuffled/{plot_params['output_name']}{plot_params['output_format']}"


    print(output_name)

    ax1.set_ylim(0.000001, 300)
    ax1.set_xlim(0.000001, 3.013)
    p13 = cosmology.Planck13
    ax1_twin = math_tools.cosmology.make_lookback_time_axis(ax1, cosmo=p13, z_range=(redshifts[0], redshifts[-1]))
    ax1_twin.set_xlabel("$\mathrm{Lookback\ Time\ [Gyr]}$")



    ax1.set_ylabel(plot_params['ylabel'])
    ax2.set_ylabel("\\textrm{Residual} \n \\textrm{(Absolute)}", fontsize=14)
    ax2.set_xlim(0.000001, 3.013)

    ax3.set_ylabel("\\textrm{Residual} \n \\textrm{(Relative \%)}", fontsize=14)
    ax3.set_xlabel("\\textrm{Redshift}")
    ax3.set_xlim(0.000001, 3.013)

    ax2.axis(ymin=-10.0, ymax=10.0, xmin=0.0001, xmax=3.013)
    ax3.axis(ymin=-4, ymax=4, xmin=0.0001, xmax=3.013)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.spines['bottom'].set_linewidth(0)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(output_name)




if __name__ == "__main__":

    sigmaCI_fit_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_fit_sigma_ci_none_uncert_expon_model_least_squares.txt"

    sigmaVar_fit_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_fit_sigma_var_none_uncert_expon_model_least_squares.txt"


    sigma_plot_params = {
        "sigmaCI_data": sigmaCI_fit_filename,
        "sigmaVar_data": sigmaVar_fit_filename,
        "output_name": "Sigma_CI_Sigma_Var_Fits_None_Uncert",
        "output_format": ".png",
        "ylabel": "$\sigma \ \mathrm{[pc\ cm^{-3}]}$",
        "sigmaCI_fit_lstyle": "-",
        "sigmaCI_fit_lwidth": 3,
        "sigmaCI_fit_lcolor": "#ca0020",
        "sigmaCI_fit_label": "$\mathrm{Fit}:\ \sigma_\mathrm{CI}\ $",
        "sigmaVar_fit_lstyle": "--",
        "sigmaVar_fit_lwidth": 3,
        "sigmaVar_fit_label": "$\mathrm{Fit}:\ \sigma_\mathrm{Var}\ $",
        "sigmaVar_fit_lcolor": "#0571b0",
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



    plot_fit(sigma_plot_params, data_params=data_params, stat_types=["sigmaCI", "sigmaVar"], simname="RefL0100N1504")
