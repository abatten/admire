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

    for model in all_models:
        output_file_name = f"./analysis_outputs/shuffled/ANALYSIS_{model.label}_mean_var_std_from_pdf.txt"


        redshifts = model.z_bins

        #mean_arr = np.zeros_like(redshifts)
        #std_arr = np.zeros_like(redshifts)
        #conf_int_sigma = np.zeros_like(redshifts)

        output = open(output_file_name, "w")

        output.write(f"Statistics of EAGLE Simulation: {model.label}\n")

        output.write(
            "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "Redshift",
            "Mean",
            "Median",
            "Variance",
            "Sigma",
            "CI Lower",
            "CI Upper",
            "CI Width",
            "CI Width/2",
            )
        )


        for zidx, z in enumerate(redshifts):

            dm_pdf = model.Hist.T[zidx]
            dm_bins = model.DM_bin_centres#[1:]
            dm_bin_widths = model.DM_bin_widths

            mean = calc_mean_from_pdf(dm_bins, dm_pdf, dx=dm_bin_widths)

            conf_int_l = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std_lim_l)
            conf_int_u = calc_val_from_pdf_percentile(dm_bins, dm_pdf, std_lim_u)

            conf_int_width = conf_int_u - conf_int_l
            conf_int_width_2 = 0.5 * conf_int_width
            conf_int_sigma = conf_int_width_2
            median = calc_median_from_pdf(dm_bins, dm_pdf)
            std = calc_std_from_pdf(dm_bins, dm_pdf, dx=dm_bin_widths)
            var = calc_variance_from_pdf(dm_bins, dm_pdf, dx=dm_bin_widths)

        #per_arr = std_arr / mean_arr


            output.write(
                f"{z:<10.3f} {mean:<10.3f} {median:<10.3f} "
                f"{var:<10.3f} {std:<10.3f} {conf_int_l:<10.3f} "
                f"{conf_int_u:<10.3f} {conf_int_width:<10.3f} "
                f"{conf_int_width/2:<10.3f} "
                " \n"
            )





        output.close()

        #ax.plot(redshifts, mean_arr, label=model.label)
        #plt.clf()
     #   plt.plot(redshifts, std_arr**2, label=model.label)
        #plt.savefig("STD_V_REDSHIFTS2.png")
        #plt.clf()
        #plt.plot(redshifts, conf_int_sigma, label=model.label)
        #plt.savefig("STD_MEAN_V_REDSHIFTS2.png")
        #plt.clf()
    #ax.set_ylim(50, 60)
    #ax.set_xlim(0, 0.2)
    #plt.legend(frameon=False)
    #plt.savefig("STD_V_REDSHIFTS.png")


   #  print(mean, median, std, var, std/mean, var/mean)

if __name__ == "__main__":

    print_tools.print_header("Analysis: Calculate Model Statistics")


    RefL0025N0376 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/output/T4EOS",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/shuffled_output/",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RefL0025N0376",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    RefL0025N0752 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0752/all_snapshot_data/output/T4EOS",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0752/all_snapshot_data/shuffled_output/",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RefL0025N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    RecalL0025N0752 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RecalL0025N0752/all_snapshot_data/output/T4EOS",
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RecalL0025N0752/all_snapshot_data/shuffled_output/",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RecalL0025N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    RefL0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RefL0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    NoAGNL0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_NoAGNL0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "NoAGNL0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    AGNdT9L0050N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_AGNdT9L0050N0752/all_snapshot_data/shuffled_output/",
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        #"file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "AGNdT9L0050N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    RefL0100N1504 = {
        #"dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
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

    RandL0100 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/Random_Gaussian_Maps/RandL0100/",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RandGaussL0100",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.arctic_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }

    RandL0025 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/Random_Gaussian_Maps/RandL0025/",
        "file_name"    : "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5",
        "label"        : "RandGaussL0025",
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
    #    RefL0100N1504,
    #     RefL0025N0376,
    #    RefL0025N0752,
    #   RecalL0025N0752,
    #    RefL0050N0752,
    #RandL0100,
    #RandL0025,
    NoAGNL0050N0752,
    AGNdT9L0050N0752,
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



