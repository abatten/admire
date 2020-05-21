import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import curve_fit
from scipy.integrate import quad
import scipy.stats as stats

import cmasher
import e13tools
from glob import glob


from pyx import plot_tools
from pyx import print_tools

from model import DMzModel

#the properties of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('xtick', direction="in")
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=12)
plt.rc('ytick', direction="in")
plt.rc('axes', labelsize=12)
plt.rc('axes', labelsize=12)




def gaussian(x, mu, sigma, coeff):
    """
    """
    #coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = - 0.5 * ((x - mu) / (sigma))**2

    return coeff * np.e**exponent

def log_normal(x, mu, shape, coeff):
    """
    """
    exponent = - 0.5 * ((np.log(x) - mu)**2 / shape**2)
    return (coeff) * np.e**exponent
#def log_normal(x, mu, shape):
#    """
#    """
#    exponent = - 0.5 * ((np.log(x) - mu)**2 / shape**2)
#    coeff = (x * shape * np.sqrt(2 * np.pi))**-1
#    return (coeff) * np.e**exponent


#def logistic(x, mu, shape):

def log_logistic(x, alpha, beta):
    """
    """
    top = (beta / alpha) * (x / alpha)**(beta - 1)
    bot = 1 + (x / alpha)**(-1 * beta)

    return top/bot

def skew_norm(x, eta, omega, alpha, coeff):

    #term1 = 2 / (omega * np.sqrt(2 * np.pi))

    term2 = np.e**(-0.5 * ((x - eta) / omega)**2)

    def term3_integrand(t):
        return (1 / (2 * np.pi)) * np.e**(-0.5 * t**2 )

    term3_lower_limit = -1 * np.inf

    term3 = np.zeros(len(x))
    for idx, xval in enumerate(x):
        term3_upper_limit = alpha * ((xval - eta)/omega)

        term3[idx] = quad(term3_integrand, term3_lower_limit, term3_upper_limit)[0]

    return coeff * term2 * term3

def log_norm(pdf, mu, shape):
    return stats.lognorm.pdf(pdf, s=shape, loc=mu)

def norm(pdf, mu, sigma, scale):
    return stats.norm.pdf(pdf, loc=mu, scale=sigma)

#print("Making TEST PDF")
#test_array = np.linspace(0, 10)
#test_pdf = skew_norm(test_array, 1, 1, 1)
#
#plt.plot(test_array, test_pdf)
#plt.savefig("TESTINGSKEWNORM.png")
#plt.clf()
#
#print("Finished TEST PDF")



def calc_median(pdf, bins):
    """
    Calculates the median of a pdf. Where P(X < median) = 1/2

    """

    cdf = np.cumsum(pdf)
    median_idx = np.where(cdf > 0.5)[0][0]

    median = bins[median_idx]
    return median
    





def fit_pdf_function(pdf, bins, function='log_logistic'):
    max_value = np.max(pdf)
    min_value = np.min(pdf)

    if function == 'log_logistic':
        median = calc_median(pdf, bins)
        alpha = median
        print("median: ", alpha)

        fit = curve_fit(log_logistic, bins, pdf, bounds=([0.999999999*alpha, 2], [alpha, 10]))
    
    if function == 'gaussian':
        max_range = bins[np.where(pdf > 1e-4)[0][-1]]
        min_range = bins[np.where(pdf > 1e-4)[0][0]]
        fit = curve_fit(gaussian, bins, pdf, bounds=([min_range, 0, min_value], [max_range, 400, 2 * max_value]))
        
    elif function == 'log_normal':
        max_range = bins[np.where(pdf > 1e-4)[0][-1]]
        min_range = bins[np.where(pdf > 1e-4)[0][0]]
        fit = curve_fit(log_normal, bins, pdf, bounds=([0, 0, 0], [max_range, 1000, max_value]))
        #fit = curve_fit(log_normal, bins, pdf, bounds=([0, 0], [max_range, 1000]))
    
    elif function == 'log_norm':
        max_range = bins[np.where(pdf > 1e-4)[0][-1]]
        min_range = bins[np.where(pdf > 1e-4)[0][0]]
        fit = curve_fit(log_norm, bins, pdf, bounds=([0, 0], [max_range, 1000]))
    
    elif function == 'norm':
        max_range = bins[np.where(pdf > 1e-4)[0][-1]]
        min_range = bins[np.where(pdf > 1e-4)[0][0]]
        fit = curve_fit(norm, bins, pdf, bounds=([0, 0, 0], [max_range, 1000, max_value]))

    elif function == "skew_norm":
        max_range = bins[np.where(pdf > 1e-4)[0][-1]]
        min_range = bins[np.where(pdf > 1e-4)[0][0]]
        fit = curve_fit(skew_norm, bins, pdf, bounds=([0, 0, 0, 0], [max_range, 1000, 1000, max_value]))

    return fit


def log_normal_mean(mu, sigma):
    
    exponent = mu + 0.5 * sigma**2

    return np.exp(exponent)


def log_normal_variance(mu, sigma):
    """
    Computes the variance of a log-normal distribution.
    """
    exponent1 = sigma**2
    exponent2 = 2 * mu + sigma**2

    return (np.exp(exponent1) - 1) * np.exp(exponent2)

def log_logistic_mean(alpha, beta):
    
    if beta < 1:
        raise ValueError("Beta must be greater than 1 for mean to be defined")

    top = alpha * np.pi / beta
    bot = np.sin(np.pi / beta)
    return top / bot

def log_logistic_variance(alpha, beta):
    
    if beta < 2:
        raise ValueError("Beta must be greater than 2 for variance to be defined")

    b = np.pi / beta
    var = alpha**2 * ((2 * b / np.sin(2 * b)) - (b**2 / np.sin(b)**2))
    return var


def skew_norm_mean(eta, omega, alpha):
    delta = alpha / np.sqrt(1 + alpha**2)
    return eta + omega * delta * np.sqrt(2/np.pi)

def skew_norm_variance(eta, omega, alpha):
    delta = alpha / np.sqrt(1 + alpha**2)
    return omega**2 * (1 - 2 * delta**2 / np.pi)


if __name__ == "__main__":
    print_tools.print_header("Fitting DM-z Relation")

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
    
    RefL0100N1504 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5",
        "label"        : "RefL0100N1504",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : 'blue',
        "linestyle"    : ':',
        "linewidth"    : 2,
        "marker"       : None,
        "plot_toggle"  : True,
    }
    

    model_dicts = [
    #    RefL0025N0752,
    #    RefL0025N0376,
    #    RecalL0025N0752
        RefL0100N1504,
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

    # Only choose one model at a time
    model = all_models[0]

    DM_bin_centres = model.DM_bin_centres
    DM_bin_edges = model.DM_bins
    z_vals = model.z_bins

    num_cols = 6
    num_rows = 11


    fig, ax = plt.subplots(ncols=num_cols, nrows=num_rows, constrained_layout=True, sharey=True, figsize=(16, 32))

    #output = open("RefL0100N1504_Log_Normal_Fit_Mean_Std.txt", "w")
    #output.write("{:<8} {:<8} {:<8} {:<8} \n".format("Redshift", "Mean", "Std", "Percent"))

    FIT_TYPE = "norm"

    # Open file and write header
    output_filename = f"{model.label}_{FIT_TYPE}_fit_mean_std_percent.txt"
    output = open(output_filename, "w")
    output.write("{:<8} {:<8} {:<8} {:<8} \n".format("Redshift", "Mean", "Std", "Percent"))


    plot_num = 0
    # Turn axes into 1D array to loop over.
    axes = ax.ravel()

    for idx, pdf in enumerate(all_models[0].Hist.T):
        print(idx)
        redshift = z_vals[idx]

        fit = fit_pdf_function(pdf, DM_bin_centres, function=FIT_TYPE)

        if FIT_TYPE == "log_normal":
            mu_fit, sigma_fit, coeff_fit = fit[0]


            # Calculate Stats
            mean = log_normal_mean(mu=mu_fit, sigma=sigma_fit)
            std = np.sqrt(log_normal_variance(mu=mu_fit, sigma=sigma_fit))
            percent = std/mean

            # Plot PDF and FIT
            axes[plot_num].plot(DM_bin_centres, pdf, color="blue", label="PDF")
            axes[plot_num].plot(DM_bin_centres, log_normal(DM_bin_centres, mu_fit, sigma_fit, coeff_fit), color="red", label="Fit")

            # Find the DM range where the PDF is non-zero
            val_range = np.where(pdf > 1e-4)
            axes[plot_num].set_xlim(DM_bin_edges[val_range[0][0]], DM_bin_edges[val_range[0][-1]])

            # Add stats data to plot
            axes[plot_num].text(0.55, 0.9, f"$z = $ {z_vals[idx]:.3f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.83, f"mean = {mean:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.76, f"std = {std:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.69, f"\% = {percent:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.62, f"Fit $\mu = $ {mu_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.55, f"Fit $\sigma = $ {sigma_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].legend(loc='upper left', bbox_to_anchor=(0.50, 0.55), frameon=False, fontsize=10)
        
        if FIT_TYPE == "log_norm":
            mu_fit, sigma_fit = fit[0]

            # Calculate Stats
            mean, var = stats.lognorm.stats(s=sigma_fit, loc=mu_fit, moments='mv')
            std = np.sqrt(var)
            percent = std/mean

            # Plot PDF and FIT
            axes[plot_num].plot(DM_bin_centres, pdf, color="blue", label="PDF")
            axes[plot_num].plot(DM_bin_centres, stats.lognorm.pdf(DM_bin_centres, s=sigma_fit, loc=mu_fit), color="red", label="Fit")

            # Find the DM range where the PDF is non-zero
            val_range = np.where(pdf > 1e-4)
            axes[plot_num].set_xlim(DM_bin_edges[val_range[0][0]], DM_bin_edges[val_range[0][-1]])

            # Add stats data to plot
            axes[plot_num].text(0.55, 0.9, f"$z = $ {z_vals[idx]:.3f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.83, f"mean = {mean:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.76, f"std = {std:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.69, f"\% = {percent:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.62, f"Fit $\mu = $ {mu_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.55, f"Fit $\sigma = $ {sigma_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].legend(loc='upper left', bbox_to_anchor=(0.50, 0.55), frameon=False, fontsize=10)
        
        if FIT_TYPE == "norm":
            mu_fit, sigma_fit = fit[0]

            # Calculate Stats
            mean, var = stats.norm.stats(loc=mu_fit, scale=sigma_fit, moments='mv')
            std = np.sqrt(var)
            percent = std/mean

            # Plot PDF and FIT
            axes[plot_num].plot(DM_bin_centres, pdf, color="blue", label="PDF")
            axes[plot_num].plot(DM_bin_centres, stats.norm.pdf(DM_bin_centres, scale=sigma_fit, loc=mu_fit), color="red", label="Fit")

            # Find the DM range where the PDF is non-zero
            val_range = np.where(pdf > 1e-4)
            axes[plot_num].set_xlim(DM_bin_edges[val_range[0][0]], DM_bin_edges[val_range[0][-1]])

            # Add stats data to plot
            axes[plot_num].text(0.55, 0.9, f"$z = $ {z_vals[idx]:.3f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.83, f"mean = {mean:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.76, f"std = {std:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.69, f"\% = {percent:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.62, f"Fit $\mu = $ {mu_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.55, f"Fit $\sigma = $ {sigma_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].legend(loc='upper left', bbox_to_anchor=(0.50, 0.55), frameon=False, fontsize=10)

        elif FIT_TYPE == "log_logistic":
            alpha_fit, beta_fit = fit[0]

            # Calculate Stats
            mean = log_logistic_mean(alpha=alpha_fit, beta=beta_fit)
            std = np.sqrt(log_logistic_variance(alpha=alpha_fit, beta=beta_fit))
            percent = std/mean

            axes[plot_num].plot(DM_bin_centres, pdf, color="blue", label="PDF")
            axes[plot_num].plot(DM_bin_centres, log_logistic(DM_bin_centres, alpha_fit, beta_fit), color="red", label="Fit")

            # Find the DM range where the PDF is non-zero
            val_range = np.where(pdf > 1e-4)
            axes[plot_num].set_xlim(DM_bin_centres[val_range[0][0]], DM_bin_centres[val_range[0][-1]])
            axes[plot_num].set_ylim(0, 0.061)

            # Add stats data to plot
            axes[plot_num].text(0.55, 0.9, f"$z = $ {z_vals[idx]:.3f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.83, f"mean = {mean:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.76, f"std = {std:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.69, f"\% = {percent:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.62, f"Fit $\\alpha = $ {alpha_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.55, f"Fit $\\beta = $ {beta_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].legend(loc='upper left', bbox_to_anchor=(0.50, 0.55), frameon=False, fontsize=10)

        elif FIT_TYPE == "skew_norm":
            eta_fit, omega_fit, alpha_fit, coeff_fit = fit[0]
            print("PDF Sum :", np.sum(pdf))
            # Calculate Stats
            mean = skew_norm_mean(eta=eta_fit, omega=omega_fit, alpha=alpha_fit)
            std = np.sqrt(skew_norm_variance(eta=eta_fit, omega=omega_fit, alpha=alpha_fit))
            percent = std/mean

            axes[plot_num].plot(DM_bin_centres, pdf, color="blue", label="PDF")
            axes[plot_num].plot(DM_bin_centres, skew_norm(DM_bin_centres, eta_fit, omega_fit, alpha_fit, coeff_fit), color="red", label="Fit")
            
            # Find the DM range where the PDF is non-zero
            val_range = np.where(pdf > 1e-4)
            axes[plot_num].set_xlim(DM_bin_centres[val_range[0][0]], DM_bin_centres[val_range[0][-1]])
            axes[plot_num].set_ylim(0, 0.061)

            # Add stats data to plot
            axes[plot_num].text(0.55, 0.9, f"$z = $ {z_vals[idx]:.3f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.83, f"mean = {mean:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.76, f"std = {std:.0f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.69, f"\% = {percent:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.62, f"Fit $\eta = $ {eta_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.55, f"Fit $\omega = $ {omega_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].text(0.55, 0.48, f"Fit $\\alpha = $ {alpha_fit:.2f}", horizontalalignment='left', verticalalignment='center', transform=axes[plot_num].transAxes, fontsize=11)
            axes[plot_num].legend(loc='upper left', bbox_to_anchor=(0.50, 0.48), frameon=False, fontsize=10)


        output.write(f"{redshift:<8.3f} {mean:<8.3f} {std:<8.3f} {percent:<8.3f} \n")


    



        #if idx % 3 == 0 or idx == 65:
        #axes[plot_num].plot(bins, log_normal(bins, fit[0][0], fit[0][1]), color="red", label="Fit")
        #axes[plot_num].plot(bins, gaussian(bins, fit[0][0], fit[0][1], fit[0][2]), color="red")
        #axes[plot_num].plot(bins, gaussian(bins, fit[0][0], fit[0][1], fit[0][2]), color="red")


        plot_num +=1


    # Add the X and Y axis labels
    for idx_row, row in enumerate(ax):
        for idx_col, cell in enumerate(row):
            if idx_col == 0:
                cell.set_ylabel("PDF")
            if idx_row == len(ax) - 1:
                cell.set_xlabel(r"$\mathrm{DM\ [pc\ cm^{-3}]}$")
            


    output.close()
    plt.savefig(f"{model.label}_{FIT_TYPE}_Fits.png", dpi=200)

    print_tools.print_footer()




