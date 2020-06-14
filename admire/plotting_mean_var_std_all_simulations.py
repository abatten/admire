import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from glob import glob
import cmasher


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
        "legend.fontsize": 16,
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


def plot_statistic(data_files, stat_type, output_format=".png"):


    colours = np.array(
        list(
            map(mpl.colors.to_hex, cmasher.rainforest(np.linspace(0.15, 0.80, 4)))

            )
        )
    #colours = ["#FFB300", "#803E75", "#FF6800", "#A6BDD7"]#, "#C10020", "#D39B85", "#817066"]
    #color_list = ["#000000", "#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020", "#D39B85", "#817066"]
    plot_style_dict = {
        "RefL0100N1504": ("#000000", 3, "-"),
        "RefL0025N0376": (colours[2], 2, ":"),
        "RefL0025N0752": (colours[1], 2, "--"),
        "RefL0050N0752": (colours[0], 2, ":"),
        "RecalL0025N0752": (colours[3], 2, "--"),

    }

    data_dict = {
        "mean" : {
            "col": 1,
            "output_name": "ALL_SIMS_mean_v_redshift",
            "ylabel": "$\langle \mathrm{DM_{cosmic}} \\rangle\ \left[\mathrm{pc\ cm^{-3}} \\right]$"
        },
        "median" : {
            "col": 2,
            "output_name": "ALL_SIMS_median_v_redshift",
            "ylabel": "$\mathrm{Median\ DM_{cosmic}}\ \left[\mathrm{pc\ cm^{-3}} \\right]$"
        },
        "variance" : {
            "col": 3,
            "output_name": "ALL_SIMS_variance_v_redshift",
            "ylabel": "$\sigma_G^2\ \left[\mathrm{pc\ cm^{-3}} \\right]$"
        },
        "sigma_g" : {
            "col": 4,
            "output_name": "ALL_SIMS_sigma_g_v_redshift",
            "ylabel": "$\sigma_\mathrm{G}\ \left[\mathrm{pc\ cm^{-3}} \\right]$"
        },
        "ci_width" : {
            "col": 7,
            "output_name": "ALL_SIMS_ci_width_v_redshift",
            "ylabel": "\\textrm{Confidence Interval Width}\ $\left[\mathrm{pc\ cm^{-3}} \\right]$"
        },
        "sigma_ci" : {
            "col": 8,
            "output_name": "ALL_SIMS_sigma_ci_v_redshift",
            "ylabel": "$\sigma_\mathrm{CI}\ \left[\mathrm{pc\ cm^{-3}} \\right]$"
        },
        "sigma_ci_sigma_g": {
            "col": None,
            "output_name": "ALL_SIMS_sigma_ci_sigma_g_v_redshift",
            "ylabel": "$\sigma_\mathrm{CI}/\sigma_\mathrm{G}$",
        },
        "sigma_ci_mean": {
            "col": None,
            "output_name": "ALL_SIMS_sigma_ci_mean_v_redshift",
            "ylabel": "$\sigma_\mathrm{CI}/\langle \mathrm{DM_{cosmic}} \\rangle$",
        },
        "sigma_g_mean": {
            "col": None,
            "output_name": "ALL_SIMS_sigma_g_mean_v_redshift",
            "ylabel": "$\sigma_\mathrm{G}/\langle \mathrm{DM_{cosmic}} \\rangle$",
        },

    }

    plt.rcParams.update(set_rc_params(usetex=True))
    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True)

    for filename in data_files:
        sim_name = filename.split("_")[2]


        data = np.loadtxt(filename, unpack=True, skiprows=2)

        redshifts = data[0]

        if isinstance(data_dict[stat_type]["col"], int):
            data_col = data_dict[stat_type]["col"]
            stat = data[data_col]
        else:
            if stat_type == "sigma_ci_sigma_g":
                stat = data[data_dict["sigma_ci"]["col"]] / data[data_dict["sigma_g"]["col"]]

            elif stat_type == "sigma_ci_mean":
                stat = data[data_dict["sigma_ci"]["col"]] / data[data_dict["mean"]["col"]]

            elif stat_type == "sigma_g_mean":
                stat = data[data_dict["sigma_g"]["col"]] / data[data_dict["mean"]["col"]]


        lcolor, lwidth, lstyle = plot_style_dict[sim_name]

        ax.plot(redshifts, stat,
            color=lcolor,
            linewidth=lwidth,
            linestyle=lstyle,
            label=f"\\textrm{{{sim_name}}}",
            )


    plt.legend()
    ax.set_xlabel("\\textrm{{Redshifts}}")
    ax.set_ylabel(data_dict[stat_type]["ylabel"])
    plt.savefig(f"analysis_plots/{data_dict[stat_type]['output_name']}{output_format}")


if __name__ == "__main__":
    all_files = sorted(glob("./analysis_outputs/*mean_var_std_from_pdf*.txt"))
    all_files.reverse()

    plot_toggle = {
        "mean": True,
        "median": True,
        "variance": True,
        "sigma_g": True,
        "sigma_ci": True,
        "ci_width": True,
        "sigma_ci_sigma_g": True,
        "sigma_ci_mean": True,
        "sigma_g_mean": True,

    }

    for key in plot_toggle:
        if plot_toggle[key]:
            plot_statistic(all_files, stat_type=key)

