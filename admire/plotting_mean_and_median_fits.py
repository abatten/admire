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



def linear_relation(x, slope, intercept):
    return slope * x + intercept

def fz_relation(z, alpha):

    fz_array = np.zeros_like(z)
    def integrand(z, OmegaL=0.693, OmegaM=0.307):
        top = 1 + z
        bot = np.sqrt(OmegaM * (1+z)**3 + OmegaL)
        return top/bot

    for idx, z in enumerate(z):
        fz_array[idx] = quad(integrand, 0, z)[0]

    return alpha * fz_array


def reviewer_model(x, A, C):
    return A * x + C * x**2


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


def plot_fit(plot_params, data_params=None, stat_type="Mean", simname="RefL0100N1504"):

    sim_idx_dict = {
        "RefL0100N1504": 0,
        "RefL0025N0376": 1,
        "RefL0025N0752": 2,
        "RecalL0025N0752": 3,
        "RefL0050N0752": 4,
        "NoAGNL0050N0752": 5,
        "AGNdT9L0050N0752": 6,
    }
    sim_idx = sim_idx_dict[simname]


    stat_type_dict = {
        "Mean": 1,
        "Median": 2,
    }

    stat_idx = stat_type_dict[stat_type]



    plt.rcParams.update(set_rc_params(usetex=True))
    fig = plt.figure(figsize=(10,7.5))
    gs = fig.add_gridspec(18, 1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[:9, 0])
    ax2 = fig.add_subplot(gs[9:12, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[12:15, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[15:18, 0], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.spines['bottom'].set_linewidth(0)


    #w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads()
    #fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)

    #fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True)
    #w_pad, h_pad, wspace, hspace = fig.get_constrained_layout_pads()
    #fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)

    linear_data = np.loadtxt(plot_params["linear_data"], unpack=True, skiprows=1, usecols=range(1,8))
    non_linear_data = np.loadtxt(plot_params["non_linear_data"], unpack=True, skiprows=1, usecols=range(1,8))
    #reviewer_data = np.loadtxt(plot_params["reviewer_data"], unpack=True, skiprows=1, usecols=range(1,8))
    sigma_data = np.loadtxt(plot_params["sigma_data"], unpack=True, skiprows=2)

    sim_data = np.loadtxt(data_params["filename"], unpack=True, skiprows=2)

    redshifts = sim_data[0]
    stat = sim_data[stat_idx]

    linear_slope, linear_intercept = linear_data[0][sim_idx], linear_data[2][sim_idx]
    lin_dm_vals = linear_relation(redshifts, linear_slope, linear_intercept)

    non_linear_alpha = non_linear_data[0][sim_idx]
    nonlin_dm_vals = fz_relation(redshifts, non_linear_alpha)

    #reviewer_A, reviewer_C = reviewer_data[0][sim_idx], reviewer_data[2][sim_idx]
    #reviewer_dm_vals = reviewer_model(redshifts, reviewer_A, reviewer_C)

    sigma = sigma_data[4]

    plot_params["lin_label"] += f": $\left\langle\mathrm{{DM_{{cosmic}}}}\\right\\rangle = {linear_slope:.1f}z + {linear_intercept:.1f}$"
    plot_params["nlin_label"] += f": $\left\langle\mathrm{{DM_{{cosmic}}}}\\right\\rangle = {non_linear_alpha:.1f}F(z)$"
    #plot_params["rev_label"] += f": $\left\langle\mathrm{{DM_{{cosmic}}}}\\right\\rangle = {reviewer_A:.1f}z {reviewer_C:.1f}z^2$"


    # Plot Actual Mean from Sim
    ax1.plot(redshifts, stat,
        color=plot_params["data_lcolor"],
        linewidth=plot_params["data_lwidth"],
        linestyle=plot_params["data_lstyle"],
        label=plot_params["data_label"])

    # Plot the Linear Model Fit
    ax1.plot(redshifts, lin_dm_vals,
        color=plot_params["lin_lcolor"],
        linewidth=plot_params["lin_lwidth"],
        linestyle=plot_params["lin_lstyle"],
        label=plot_params["lin_label"])

    # Plot the Non-Linear Model Fit
    ax1.plot(redshifts, nonlin_dm_vals,
        color=plot_params["nlin_lcolor"],
        linewidth=plot_params["nlin_lwidth"],
        linestyle=plot_params["nlin_lstyle"],
        label=plot_params["nlin_label"])

    # Plot the Reviewer Model Fit
    #ax1.plot(redshifts, reviewer_dm_vals,
    #    color=plot_params["rev_lcolor"],
    #    linewidth=plot_params["rev_lwidth"],
    #    linestyle=plot_params["rev_lstyle"],
    #    label=plot_params["rev_label"])

    # Plot the Linear Model Fit Rediduals
    ax2.plot(redshifts, stat - lin_dm_vals,
        color=plot_params["lin_lcolor"],
        linewidth=plot_params["lin_lwidth"],
        linestyle=plot_params["lin_lstyle"],
        label=plot_params["lin_label"])

    # Plot the Non-Linear Model Fit Residuals
    ax2.plot(redshifts, stat - nonlin_dm_vals,
        color=plot_params["nlin_lcolor"],
        linewidth=plot_params["nlin_lwidth"],
        linestyle=plot_params["nlin_lstyle"],
        label=plot_params["nlin_label"])

    # Plot the Reviewer Model Fit
    #ax2.plot(redshifts, stat - reviewer_dm_vals,
    #    color=plot_params["rev_lcolor"],
    #    linewidth=plot_params["rev_lwidth"],
    #    linestyle=plot_params["rev_lstyle"],
    #    label=plot_params["rev_label"])

    ax2.plot(np.linspace(-1, 4, 100), np.zeros(100), "grey", linewidth=1)
    # Plot the Linear Model Fit Relative Rediduals
    ax3.plot(redshifts, (stat - lin_dm_vals)/stat * 100,
        color=plot_params["lin_lcolor"],
        linewidth=plot_params["lin_lwidth"],
        linestyle=plot_params["lin_lstyle"],
        label=plot_params["lin_label"])

    ax3.plot(np.linspace(-1, 4, 100), np.zeros(100), "grey", linewidth=1)
    # Plot the Non-Linear Model Fit Relative Residuals
    ax3.plot(redshifts, (stat - nonlin_dm_vals)/stat * 100,
        color=plot_params["nlin_lcolor"],
        linewidth=plot_params["nlin_lwidth"],
        linestyle=plot_params["nlin_lstyle"],
        label=plot_params["nlin_label"])

    # Plot the Reviewer Model Fit
    #ax3.plot(redshifts, (stat - reviewer_dm_vals)/stat * 100,
    #    color=plot_params["rev_lcolor"],
    #    linewidth=plot_params["rev_lwidth"],
    #    linestyle=plot_params["rev_lstyle"],
    #    label=plot_params["rev_label"])

    ax4.plot(np.linspace(-1, 4, 100), np.zeros(100), "grey", linewidth=1)
    # Plot the Linear Model Fit Rediduals
    ax4.plot(redshifts, (stat - lin_dm_vals)/sigma,
        color=plot_params["lin_lcolor"],
        linewidth=plot_params["lin_lwidth"],
        linestyle=plot_params["lin_lstyle"],
        label=plot_params["lin_label"])

    # Plot the Non-Linear Model Fit Residuals
    ax4.plot(redshifts, (stat - nonlin_dm_vals)/sigma,
        color=plot_params["nlin_lcolor"],
        linewidth=plot_params["nlin_lwidth"],
        linestyle=plot_params["nlin_lstyle"],
        label=plot_params["nlin_label"])

    # Plot the Reviewer Model Fit
    #ax4.plot(redshifts, (stat - reviewer_dm_vals)/sigma,
    #    color=plot_params["rev_lcolor"],
    #    linewidth=plot_params["rev_lwidth"],
    #    linestyle=plot_params["rev_lstyle"],
    #    label=plot_params["rev_label"])


    print(f"stat: {stat_type}",  "mean linear residual", np.mean(((stat - lin_dm_vals)/stat * 100)**2)**0.5)
    print(f"stat: {stat_type}",  "mean nonlinear residual", np.mean(((stat - nonlin_dm_vals)/stat * 100)**2)**0.5 )

    if stat_type == "Mean":
        ax2.axis(ymin=-45, ymax=45, xmin=0.0001, xmax=3.013)
        ax3.axis(ymin=-10, ymax=10, xmin=0.0001, xmax=3.013)
    elif stat_type == "Median":
        ax2.axis(ymin=-45, ymax=45, xmin=0.0001, xmax=3.013)
        ax3.axis(ymin=-20, ymax=20, xmin=0.0001, xmax=3.013)


    output_name = \
        f"./analysis_plots/shuffled/{simname}_{plot_params['output_name']}{plot_params['output_format']}"


    print(output_name)

    ax1.set_ylim(0.000001, 3100)
    ax1.set_xlim(0.000001, 3.013)
    p13 = cosmology.Planck13
    ax1_twin = math_tools.cosmology.make_lookback_time_axis(ax1, cosmo=p13, z_range=(redshifts[0], redshifts[-1]))
    ax1_twin.set_xlabel("$\mathrm{Lookback\ Time\ [Gyr]}$")



    ax1.set_ylabel(plot_params['ylabel'])
    ax2.set_ylabel("\\textrm{Residual} \n \\textrm{(Absolute)}", fontsize=14)
    ax2.set_xlim(0.000001, 3.013)

    ax3.set_ylabel("\\textrm{Residual} \n \\textrm{(Relative \%)}", fontsize=14)
    ax3.set_xlim(0.000001, 3.013)

    ax4.set_ylabel("$\\frac{\mathrm{Residual}}{\sigma_\mathrm{Var}}$", fontsize=14)
    ax4.set_xlabel("\\textrm{Redshift}")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax1.spines['bottom'].set_linewidth(0)
    ax1.legend()
    plt.savefig(output_name, dpi=200)




if __name__ == "__main__":

    mean_linear_fit_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_fit_mean_none_uncert_linear_model_least_squares.txt"

    mean_non_linear_fit_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_fit_mean_none_uncert_non_linear_model_least_squares.txt"

    mean_reviewer_fit_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_fit_mean_none_uncert_reviewer_model_least_squares.txt"

    sigma_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_RefL0100N1504_mean_var_std_from_pdf.txt"

    median_linear_fit_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_fit_median_none_uncert_linear_model_least_squares.txt"

    median_non_linear_fit_filename = \
        "./analysis_outputs/shuffled/ANALYSIS_fit_median_none_uncert_non_linear_model_least_squares.txt"


    mean_plot_params = {
        "linear_data": mean_linear_fit_filename,
        "non_linear_data": mean_non_linear_fit_filename,
        "reviewer_data": mean_reviewer_fit_filename,
        "sigma_data": sigma_filename,
        "output_name": "REVIEWER_TEST_Mean_DM_Linear_Non_Linear_Fits_None_Uncert",
        "output_format": ".png",
        "ylabel": "$\mathrm{\langle DM_{cosmic}\\rangle \ [pc\ cm^{-3}] }$",
        "lin_lstyle": ":",
        "lin_lwidth": 3,
        "lin_lcolor": "#ca0020",
        "lin_label": "\\textrm{Model A Fit}",
        "nlin_lstyle": "--",
        "nlin_lwidth": 3,
        "nlin_label": "\\textrm{Model B Fit}",
        "nlin_lcolor": "#0571b0",
        "rev_lstyle": "-.",
        "rev_lwidth": 3,
        "rev_label": "\\textrm{New Fit}",
        "rev_lcolor": "#00FF00",
        "data_lstyle": "-",
        "data_lwidth": 4,
        "data_lcolor": "#000000",
        "data_label": "$\mathrm{This\ Work\ } \left\langle\mathrm{DM_{cosmic}}\\right\\rangle$",

    }


    median_plot_params = {
        "linear_data": median_linear_fit_filename,
        "non_linear_data": median_non_linear_fit_filename,
        "output_name": "Median_DM_Linear_Non_Linear_Fits_None_Uncert",
        "output_format": ".png",
        "ylabel": "$\mathrm{Median\ DM_{cosmic} \ [pc\ cm^{-3}] }$",
        "lin_lstyle": ":",
        "lin_lwidth": 3,
        "lin_lcolor": "#ca0020",
        "lin_label": "\\textrm{Model A Fit}",
        "nlin_lstyle": "--",
        "nlin_lwidth": 3,
        "nlin_label": "\\textrm{Model B Fit}",
        "nlin_lcolor": "#0571b0",
        "data_lstyle": "-",
        "data_lwidth": 4,
        "data_lcolor": "#000000",
        "data_label": "\\textrm{This Work Median}",

    }


    simulation = "RefL0100N1504"

    data_params = {
        "filename": f"./analysis_outputs/shuffled/ANALYSIS_{simulation}_mean_var_std_from_pdf.txt"
    }

    plot_fit(mean_plot_params, data_params=data_params, stat_type="Mean", simname=simulation)
    plot_fit(median_plot_params, data_params=data_params, stat_type="Median", simname=simulation)