import numpy as np
import scipy.stats as stats
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from utils import wlog

def gaussian(data, logfile=None, verbose=False):
    
    if logfile:
        wlog("Fitting: Gaussian", logfile, verbose, u=True)

    fit = stats.norm.fit(data)

    if logfile:
        wlog("Completed fit", logfile, verbose)
        wlog("Performing bootstrap to estimate error in fit", logfile, verbose)

    rand_context = np.random.randint(0, 1e7)
    bootnum = 1000

    with NumpyRNGContext(rand_context):
        if logfile:
            wlog("Running Bootstrap", logfile, verbose, u=True)
            wlog("Bootstrap Parameters:", logfile, verbose)
            wlog("bootnum: {0}".format(bootnum), logfile, verbose)
            wlog("NumpyRNGContext: {0}".format(rand_context), logfile, verbose)

    boot_resample = bootstrap(data, bootnum=bootnum, num_samples=bootnum)

    bootstrap_mean = []
    bootstrap_std = []

    for i in range(len(boot_resample)):
        resample_fit = stats.norm.fit(boot_resample[i])
        bootstrap_mean.append(resample_fit[0])
        bootstrap_std.append(resample_fit[1])

    err = (stats.norm.fit(bootstrap_mean)[1], 
           stats.norm.fit(bootstrap_std)[1])

    if logfile:
        wlog("Completed Bootstrap Analysis", logfile, verbose, u=True)
        wlog("{0:<15}{1:<15}{2:<15}".format("Parameter", "Fit", "BootUncert"), logfile, verbose)
        wlog("{0:<15}{1:<15.8}{2:<15.8}".format("Mean", fit[0], err[0]), logfile, verbose)
        wlog("{0:<15}{1:<15.8}{2:<15.8}".format("Std", fit[1], err[1]), logfile, verbose)


    return fit, err


def lognormal(data, logfile=None, verbose=False, boot=True):

    if logfile:
        wlog("Fitting: Log-Normal", logfile, verbose, u=True)

    fit = stats.lognorm.fit(data, floc=0)

    if logfile:
        wlog("Completed fit", logfile, verbose)

    if boot:
        wlog("Performing bootstrap to estimate error in fit", logfile, verbose)

        rand_context = np.random.randint(0, 1e7)
        bootnum = 1000

        with NumpyRNGContext(rand_context):
            if logfile:
                wlog("Running Bootstrap", logfile, verbose, u=True)
                wlog("Bootstrap Parameters:", logfile, verbose)
                wlog("bootnum: {0}".format(bootnum), logfile, verbose)
                wlog("NumpyRNGContext: {0}".format(rand_context), logfile, verbose)

        boot_resample = bootstrap(data, bootnum=bootnum, num_samples=bootnum)

        bootstrap_shape = []
        bootstrap_loc = []
        bootstrap_scale = []

        for i in range(len(boot_resample)):
            resample_fit = stats.lognorm.fit(boot_resample[i])
            bootstrap_shape.append(resample_fit[0])
            bootstrap_loc.append(resample_fit[1])
            bootstrap_scale.append(resample_fit[2])

            err = (stats.norm.fit(bootstrap_shape)[1], 
                   stats.norm.fit(bootstrap_loc)[1],
                   stats.norm.fit(bootstrap_scale)[1])
    if not boot:
        wlog("Did not perform bootstrap analysis for errors", logfile, verbose)
        err = ["NaN","NaN","NaN"]

    if logfile:
        wlog("Completed Bootstrap Analysis", logfile, verbose, u=True)
        wlog("{0:<15}{1:<15}{2:<15}".format("Parameter", "Fit", "BootUncert"), logfile, verbose)
        wlog("{0:<15}{1:<15.8}{2:<15.8}".format("shape", fit[0], err[0]), logfile, verbose)
        wlog("{0:<15}{1:<15}{2:<15}".format("loc", fit[1], err[1]), logfile, verbose)
        wlog("{0:<15}{1:<15.8}{2:<15.8}".format("scale", fit[2], err[2]), logfile, verbose)


    return fit, err

