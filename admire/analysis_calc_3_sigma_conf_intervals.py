import numpy as np

from scipy import interpolate
from model import DMzModel

import os
import cmasher
from pyx import print_tools

def calc_mean_from_pdf(x, pdf, dx=None):
    if dx is None:
        # If no dx is provided assume they are linearly spaced
        dx = (x[-1] - x[0]) / len(x)

    return np.sum(pdf * x * dx)

def calc_variance_from_pdf(x, pdf, dx=None):
    if dx is None:
        # If no dx is provided assume they are linearly spaced
        dx = (x[-1] - x[0]) / len(x)

    mean = calc_mean_from_pdf(x, pdf, dx)

    return np.sum(pdf * dx * (x - mean)**2)

def calc_std_from_pdf(x, pdf, dx=None):
    if dx is None:
        # If no dx is provided assume they are linearly spaced
        dx = (x[-1] - x[0]) / len(x)

    return np.sqrt(calc_variance_from_pdf(x, pdf, dx))

def calc_val_from_pdf_percentile(x, pdf, percentile):
    cumsum = np.cumsum(pdf)
    normed_cumsum = cumsum / cumsum[-1]
    interpolated_cumsum = interpolate.interp1d(normed_cumsum, x)

    return interpolated_cumsum(percentile)

def calc_median_from_pdf(x, pdf):
    return calc_val_from_pdf_percentile(x, pdf, percentile=0.5)

def normalise_to_pdf(hist, bin_widths):
    if np.sum(hist) < 1e-16:
        pdf = np.zeros(len(hist))
    else:
        pdf = hist/bin_widths/np.sum(hist)

    return pdf

def linear_interpolate_pdfs(sample, xvals, pdfs):
    x1, x2 = xvals
    pdf1, pdf2 = pdfs

    grad = (pdf2 - pdf1) / (x2 - x1)
    dist = sample - x1
    return grad * dist + pdf1

def sigma_to_pdf_percentiles(sigma):
    std = int(sigma)
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

    return std_limits[std]





def calc_stats_from_model(all_models):


    std_lim_l, std_lim_u = sigma_to_pdf_percentiles(1)
    std2_lim_l, std2_lim_u = sigma_to_pdf_percentiles(2)
    std3_lim_l, std3_lim_u = sigma_to_pdf_percentiles(3)

    for model in all_models:
        output_file_name = f"./analysis_outputs/shuffled/ANALYSIS_{model.label}_confidence_intervals.txt"
        redshifts = model.z_bins

        output = open(output_file_name, "w")

        output.write(f"Confidence Intervals of EAGLE Simulation: {model.label}\n")

        output.write(
            "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "Redshift",
            "Mean",
            "Median",
            "CI Lower",
            "CI Upper",
            "CI2 Lower",
            "CI2 Upper",
            "CI3 Lower",
            "CI3 Upper",
            )
        )

        for zidx, z in enumerate(redshifts):

            dm_pdf = model.Hist.T[zidx]
            dm_bins = model.DM_bin_centres
            dm_bin_widths = model.DM_bin_widths

            mean = calc_mean_from_pdf(dm_bins, dm_pdf, dx=dm_bin_widths)
            median = calc_median_from_pdf(dm_bins, dm_pdf)

            conf_int_l = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std_lim_l)
            conf_int_u = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std_lim_u)

            conf_int_l_2 = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std2_lim_l)
            conf_int_u_2 = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std2_lim_u)

            conf_int_l_3 = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std3_lim_l)
            conf_int_u_3 = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std3_lim_u)

            output.write(
                f"{z:<10.3f} {mean:<10.3f} {median:<10.3f} "
                f"{conf_int_l:<10.3f} {conf_int_u:<10.3f} "
                f"{conf_int_l_2:<10.3f} {conf_int_u_2:<10.3f} "
                f"{conf_int_l_3:<10.3f} {conf_int_u_3:<10.3f} "
                " \n"
            )
        output.close()


if __name__ == "__main__":

    print_tools.print_header("Analysis: 3 Sigma Confidence Intervals")

    L0100N1504 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/output/T4EOS",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/shuffled_output/",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RefL0100N1504",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    model_dicts = [
        L0100N1504,
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

    calc_stats_from_model(all_models)

    print_tools.print_footer()