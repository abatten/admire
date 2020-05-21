import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyx import print_tools, param_tools, math_tools
import h5py

import fast_histogram
from tqdm import tqdm
from scipy.optimize import curve_fit

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

def log_normal(x, mu, shape):
    exponent = - 0.5 * ((np.log(x) - mu)**2 / shape**2)
    coeff = (x * shape * np.sqrt(2 * np.pi))**-1
    return (coeff) * np.e**exponent



def fit_log_normal(redshift, pdf, bins):
    max_range = bins[np.where(pdf > 0e-4)[0][-1]]
    min_range = bins[np.where(pdf > 1e-4)[0][0]]
    
    # Calculate the initial mean and var guesses
    if redshift <= 1e-10: # If redshift if small set mean DM to zero.
        initial_mean = 1e-16
    else:
        initial_mean = 1000 * redshift

    initial_var = (-206*np.e**(-1.08*redshift) + 221)**2
   
    # Convert the initial mean and var to mu and sigma for lognormal
    var_mean2_param = initial_var/initial_mean**2

    initial_sigma = np.sqrt(np.log(var_mean2_param + 1))
    initial_mu = np.log(initial_mean) - 0.5 * initial_sigma
    
    initial_guess = (initial_mu, initial_sigma)
    fit = curve_fit(log_normal, bins, pdf, bounds=([-100, 0], [10, 10]), p0=initial_guess)
 
    return fit






def run(params):

    # Create the file paths for the master file and the output file.
    masterfile = os.path.join(params["datadir"], params["masterfilename"] + ".hdf5")
    outputfile = os.path.join(params["outputdir"], params["outputfilename"])
    
    # Open the master and output file
    with h5py.File(masterfile, mode="r") as master, open(outputfile, mode='w') as output:
        num_slices = params["num_slices"]
        redshifts = np.zeros(num_slices)
        variances = np.zeros(num_slices)
        means = np.zeros(num_slices)
        stds = np.zeros(num_slices)
        lg_norm_means = np.zeros(num_slices)
        lg_norm_vars = np.zeros(num_slices)
        lg_norm_stds = np.zeros(num_slices)

        hist_list = []
        
        pbar = tqdm(range(num_slices))
        for index in pbar:
            pbar.set_description(f"Slice: {index}")
            data = master["dm_maps"][f"slice_{index:03d}"]["DM"]
            redshifts[index] = master["dm_maps"][f"slice_{index:03d}"]["HEADER"].attrs["Redshift"] + math_tools.cosmology.cMpc_to_z(50)
            variances[index] = np.var(data)
            means[index] = np.mean(data)
            stds[index] = np.std(data)


            bins = 10**np.linspace(0, 5, 1001)
            hist, edges = np.histogram(data, bins=bins, density=True)
            bin_centres = math_tools.calc_bin_centre(edges)
            fit = fit_log_normal(redshifts[index], hist, bin_centres)
            lg_norm_means[index] = log_normal_mean(fit[0][0], fit[0][1])
            lg_norm_vars[index] = log_normal_variance(fit[0][0], fit[0][1])
            lg_norm_stds[index] = np.sqrt(lg_norm_vars[index])
            
        
        var_cumsum = np.cumsum(variances)
        std_cumsum = np.sqrt(var_cumsum)
        fit_var_cumsum = np.cumsum(lg_norm_vars)
        fit_std_cumsum = np.sqrt(fit_var_cumsum)
        print(redshifts)
        print(variances)
        print(means)
        print(stds)
        output.write("{:<12s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s}".format(
            "Redshift", "Variance", "Mean", "Std", "Var_CumSum", "CumSum_Std", 
            "Fit_mean", "Fit_Var", "Fit_Std", "F_Var_CumSum", "F_CumSum_Std""\n"))

        for z, var, mu, std, varcs, svarcs, fvar, fmean, fstd, fvarcs, fstdcs in zip(redshifts, variances, means, stds, var_cumsum, std_cumsum, 
                                                                             lg_norm_vars, lg_norm_means, lg_norm_stds, fit_var_cumsum, fit_std_cumsum):
            output.write(f"{z:<12.4f} {var:<12.4f} {mu:<12.4f} {std:<12.4f} {varcs:<12.4f} {svarcs:<12.4f} {fvar:<12.4f} {fmean:<12.4f} {fstd:<12.4f} {fvarcs:<12.4f} {fstdcs:<12.4f}\n")




if __name__ == "__main__":
    print_tools.script_info.print_header("ADMIRE PIPELINE")

    if len(sys.argv) == 2:
        params = param_tools.dictconfig.read(sys.argv[1], "VarianceCorrelation")

    elif len(sys.argv) == 1:
        print("Please provide parameter file")
        sys.exit(1)

    else:
        print("Too many command line arguments!")
        sys.exit(1)

    for key, value in params.items():
        print(f"{key:<16}: {value}")
    run(params)

    print_tools.script_info.print_footer()


