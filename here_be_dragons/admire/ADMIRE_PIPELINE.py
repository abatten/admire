import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import configparser as CP
import yt
import h5py
import time
import random
import utils
from utils import wlog
import interpolate
import mirrot
import plot
import fit
from scipy import stats
import json
import glob

plt.rcParams['text.usetex'] = True


def read_params(param_path):
    """
    Read the supplied parameter file

    Parameters
    ----------
    param_path : string
        The path to the input parameter file.

    Returns
    -------
    params : dictionary
        The values from the parameter file
    """
    params = {}

    params["ParamPath"] = param_path
    params["ParamName"] = param_path.split("/")[-1]

    config = CP.ConfigParser()
    config.read(params["ParamPath"])

    params["Verbose"] = config.getboolean("General", "Verbose")
#    params["AutoRun"] = config.getboolean("General", "AutoRun")
#    params["ProgressBar"] = config.getboolean("General", "ProgressBar")

    params["ColDensMapDir"] = config.get("Data", "ColDensMapDir")
    params["OutputDataDir"] = config.get("Data", "OutputDataDir")
#    params["ProjDir"] = config.get("Data", "ProjDir")
    params["ColDensMapSuffix"] = config.get("Data", "ColDensMapSuffix")
    params["ColDensMapZVals"] = json.loads(config.get("Data", "ColDensMapZVals"))

    params["InterpMaps"] = config.getboolean("Interpolate", "InterpMaps")
    params["InterpFileName"] = config.get("Interpolate", "InterpFileName")

    params["RedshiftMin"] = config.getfloat("Analysis", "RedshiftMin")
    params["RedshiftMax"] = config.getfloat("Analysis", "RedshiftMax")
    params["DistSpacing"] = config.getfloat("Analysis", "DistSpacing")
    params["DistUnits"] = config.get("Analysis", "DistUnits")


    params["PerformStacking"] = config.getboolean("Stacking", "PerformStacking")
    params["StackFileName"] = config.get("Stacking", "StackFileName")
    params["StackMethod"] = config.get("Stacking", "StackMethod")

    params["PlotDir"] = config.get("Plot", "PlotDir")
    params["MatplotlibBackend"] = config.get("Plot", "MatplotlibBackend")

    params["CreateColDensMap"] = config.getboolean("Plot", "CreateColDensMap")
    params["ColDensMapFileName"] = config.get("Plot", "ColDensMapFileName")
    params["ColDensCmap"] = config.get("Plot", "ColDensCmap")
    params["ColDensVmax"] = config.getfloat("Plot", "ColDensVmax")
    params["ColDensVmin"] = config.getfloat("Plot", "ColDensVmin")

    params["CreateDMPdf"] = config.getboolean("Plot", "CreateDMPdf")
    params["DMPdfFileName"] = config.get("Plot", "DMPdfFileName")
    params["DMPdfBinNum"] = config.getint("Plot", "DMPdfBinNum")
    params["DMPdfBinMin"] = config.getfloat("Plot", "DMPdfBinMin")
    params["DMPdfBinMax"] = config.getfloat("Plot", "DMPdfBinMax")

    params["CreateMastHist"] = config.getboolean("Plot", "CreateMastHist")
    params["MastHistFileName"] = config.get("Plot", "MastHistFileName")

    params["FitLogNorm"] = config.getboolean("Fit", "FitLogNorm")
    params["PerformBootstrap"] = config.getboolean("Fit", "PerformBootstrap")

    params["LogDir"] = config.get("Log", "LogDir")
    params["LogFileName"] = config.get("Log", "LogFileName")
    params["YTLogLevel"] = config.getint("Log", "YTLogLevel")

    params["CreateMastFile"] = config.getboolean("MasterFile", "CreateMastFile")
    params["MastFileName"] = config.get("MasterFile", "MastFileName")

    return params


def initialise(param_path):
    """
    Initialise the pipeline by reading the parameter file and set
    up the log file

    Parameters
    ----------
    param_path : string
        The path to the parameter file for the script

    Returns
    -------
    params : dictionary
        A dictionary with the values from the parameter file.
    log :
        The log file
    """

    # First read the parameter file
    params = read_params(param_path)

    # Create the log file for script output
    log = utils.create_log_file(params["LogDir"],
                                params["LogFileName"],
                                params["Verbose"])

    # Write the used parameters to the log file
    wlog("Input Parameters", log, params["Verbose"], t=True)

    for key in list(params.keys()):
        wlog("{0:<25} {1}".format(key, params[key]), log, params["Verbose"])

    return params, log


def get_interp_data(z_sample, maps, logfile=None, verbose=True):
    """
    Finds the two projections with redshifts that are the nearest higher and
    nearest lower redshifts and extracts the

    Parameters
    ----------
    z_sample : float
        The redshift of interest in the interpolation.

    maps : array or array-like
        The filenames of the column density maps. These have the same indexing
        as redshift_arr

    logfile :
        The file to write the logs.


    Returns
    -------
    map_high : str

    map_high : str

    """

    z_exist = np.empty(len(maps))
    for i in range(len(maps)):
        with h5py.File(maps[i], "r") as ds:
            z_exist[i] = ds["Header"].attrs["Redshift"]

    # Get index
    idx_low, idx_high = utils.get_idx(z_sample, z_exist)

    # Get redshift of maps lower/higher than z_sample
    z_low, z_high = z_exist[idx_low], z_exist[idx_high]
    dist_low, dist_high = utils.z_to_mpc(z_low), utils.z_to_mpc(z_high)
    map_low, map_high = maps[idx_low], maps[idx_high]
    #data_low, data_high = h5py.File(map_low), h5py.File(map_high)

    if logfile:
        wlog("{0:<10} {1:>10}\
             {2:<10} {3:>10}".format("idx_low", idx_low,
                                     "idx_high", idx_high),
                                      logfile, verbose)
        wlog("{0:<10} {1:>10.5}\
             {2:<10} {3:>10.5}".format("z_low", z_low,
                                       "z_high", z_high),
                                       logfile, verbose)
        wlog("{0:<10} {1:>9.5}\
             {2} {3:>9.5}".format("dist_low", dist_low,
                                       "dist_high", dist_high),
                                       logfile, verbose)
        wlog("{0:<10} {1}\
             {2:<10} {3}\n".format("map_low", map_low,
                                   "map_high", map_high),
                                    logfile, verbose)

    return map_low, map_high


def create_interp_maps(z_samples, maps, params, logfile=None, verbose=False):
    """
    Create interpolated column density Maps at specific redshifts from existing
    column density maps.
    """

    for z in z_samples:
        wlog("Getting Interpolation Data for z = {0:.5f}".format(z), logfile,
                                                                verbose, u=True)

        map_low, map_high = get_interp_data(z, maps, logfile=logfile)

        interp = interpolate.linear_interp_2D(z, map_low, map_high, logfile)

        # Create a plot of the column density of the interpolated map
        if params["CreateColDensMap"]:
            if logfile:
                wlog("Plotting: Column Density Map", logfile, verbose)
            plot.coldens_map(interp, z, params)

        output_file = os.path.join(params["OutputDataDir"],
                                   "{0}_z_{1:.3f}.h5".format(params["InterpFileName"], z))

        with h5py.File(output_file, "w") as ds:
            if logfile:
                wlog("Saving: Interpolated Map", logfile, verbose)
                wlog("File Name: {0}".format(output_file), logfile, verbose)

            utils.create_dm_dataset(ds, interp)
            utils.create_redshift_attrs(ds, np.array([z]))

    return None


def reset_mirrot_options():

    mirrot_options = {'r90': (0, 1),
                      'r180': (0, 2),
                      'r270': (0, 3),
                      'm': (1, 0),
                      'mr90': (1, 1),
                      'mr180': (1, 2),
                      'mr270': (1, 3)
                      }

    return mirrot_options


if __name__ == "__main__":

    # The argument to the script is the parameter file
    if len(sys.argv) >= 2:
        if os.path.exists(sys.argv[1]):
            params, log = initialise(sys.argv[1])
        else:
            # Raise error if the parameter file does not exist
            raise OSError("Could not find file: {0}".format(sys.argv[1]))

    else:
        raise OSError("Parameter File not Supplied")

    # Set the verbose level of the script
    yt.mylog.setLevel(params["YTLogLevel"])
    verb = params["Verbose"]

    wlog("Reading Column Density Map Data", log, verb, t=True)

    # Column Density Maps ata format types. The column density maps can be in
    # either .npz (numpy) or .h5 (hdf5) file formats.
    npz_format = [".npz", "npz"]
    h5_format = [".h5", "h5"]

    # Need to speficy the redshifts of the npz files
    ColDensMapsZVals = params["ColDensMapZVals"]

    # If the Column Density Maps are .npz files convert them to .h5 files
    if params["ColDensMapSuffix"] in npz_format:
        wlog("Column Density Maps in .npz format", log, verb)
        wlog("Converting files to .h5 files", log, verb)
        ColDensMaps = utils.glob_files(params["ColDensMapDir"],
                                        "*" + params["ColDensMapSuffix"])

        for i in range(len(ColDensMaps)):
            old_fn = ColDensMaps[i].split(".npz")[0].split("/")[-1]
            new_fn = "{0}/{1}.h5".format(params["OutputDataDir"], old_fn)
            redshift = np.array([ColDensMapsZVals[i]])

            wlog("Converting {0} to .h5 format".format(old_fn), log, verb)
            utils.convert_npz_to_h5(ColDensMaps[i], new_fn, z=redshift)
            wlog("Created: {0}".format(new_fn), log, verb)

            # Change file path to the path of the new .h5 file
            ColDensMaps[i] = new_fn

    elif params["ColDensMapSuffix"] in h5_format:
        wlog("Column Density Maps in .h5 format", log, verb)
        ColDensMaps = utils.glob_files(params["ColDensMapDir"],
                                       "*" + params["ColDensMapSuffix"])

    else:
        raise TypeError("Unrecognised format type. ColDensMaps should be in\
                        .npz or .h5 format.")
        sys.exit(1)


    # Read the redshifts from the Column Density Maps
    wlog("Reading redshifts of projections", log, verb, u=True)
    wlog("{0:<10} {1:<50}".format("Redshift", "Filename"), log, verb)
    for m in ColDensMaps:
        with h5py.File(m, "r") as ds:
            z = ds["Header"].attrs["Redshift"][0]
            fn = m.split("/")[-1]
            wlog("{0:<10.5f} {1:<50}".format(z, fn), log, verb)

    # #############
    # INTERPOLATION
    # #############
    # If params["InterpMap"] = True, perform the linear interpolation of the
    # column density maps to produce maps at regular distance spacing as
    # specified by params["DistSpacing"]

    wlog("Column Density Map Interpolation", log, verb, t=True)

    if params["InterpMaps"]:
        wlog("Calculating the redshifts needed for interpolation", log, verb, u=True)
        wlog("Min/Max Redshifts: {0} / {1}".format(params["RedshiftMin"],
                                                     params["RedshiftMax"]),
                                                     log, verb)
        wlog("Distance Spacing: {0} {1}".format(params["DistSpacing"],
                                                params["DistUnits"]),
                                                log, verb)

        # Calculate the list of redshifts to interpolate from the Min/Max Redshift
        # and the DistSpacing
        redshifts_interp = utils.get_redshifts_with_dist_spacing(params["RedshiftMin"],
                                                              params["RedshiftMax"],
                                                              params["DistSpacing"])
        wlog("Number of Redshifts: {0}".format(redshifts_interp.size), log, verb)
        wlog(redshifts_interp, log, verb)

        # Create the interpolated projections, then plot and save them to .h5 files
        create_interp_maps(redshifts_interp, ColDensMaps, params, log, verb)

        dm_maps = utils.glob_files(params["OutputDataDir"], params["InterpFileName"] + "*")

    elif not params["InterpMaps"]:
        wlog("Skipping Interpolation (InterpMaps : False)", log, verb)
        dm_maps = utils.glob_files(params["ColDensMapDir"],
                                   params["InterpFileName"] + "*")

    ##############
    # MAP STACKING
    ##############
    # If params["PerformStacking"] = True, perform a stacking of the DM maps
    # over the redshifts to create DM maps for total DM at a given redshift

    wlog("Column Density Map Stacking", log, verb, t=True)
    if params["PerformStacking"]:
        wlog("Stacking Method: {0}".format(params["StackMethod"]), log, verb)

        # stack_total is the total number of maps combined so far
        stack_total = 0
        # stack_count is the number of maps stacked since mirrot was reset
        stack_count = 0

        for i in range(len(dm_maps)):
            wlog("Stacking {0}".format(dm_maps[i]), log, verb, u=True)
            with h5py.File(dm_maps[i], "r+") as ds:
                z_map = ds["Header"].attrs["Redshift"][0]
                wlog("Map Redshift: {0}".format(z_map), log, verb)

                # Create the DM PDF for the Interp Map
                if params["CreateDMPdf"]:
                    wlog("Plotting: DM PDF", log, verb, n=True)

                    binnum = params["DMPdfBinNum"]
                    bins = np.linspace(params["DMPdfBinMin"],
                                       params["DMPdfBinMax"],
                                       params["DMPdfBinNum"])
                    wlog("{0} {1} {2}".format("BinNum", "BinMin", "BinMax"),
                        log, verb)
                    wlog("{0:<6} {1:<6} {2:<6}".format(binnum, bins[0], bins[-1]),
                        log, verb)

                    data1d = utils.reshape_2D_to_1D(ds["DM"], log, verb)
                    plotfn = plot.create_dm_pdf(z_map, data1d, bins, params,
                                                fit=params["FitLogNorm"])
                    wlog("Created: {0}".format(plotfn), log, verb)

                if params["StackMethod"] == "Random":
                    stack_total += 1
                    stack_count += 1
                    wlog("Performing Random Stack", log, verb, n=True)
                    wlog("{0:<12} {1:<12}".format("Stack Count",
                                                  "Stack Total"), log, verb)
                    wlog("{0:<12} {1:<12}".format(stack_count,
                                                  stack_total), log, verb)

                    # For random stack we reset the mirrot options every time
                    mirrot_opts = reset_mirrot_options()
                    mir, rot = mirrot_opts.pop(random.choice(list(mirrot_opts.keys())))
                    wlog("Mirror: {0}".format(bool(mir)), log, verb)
                    wlog("Rotate: {0} Degrees (CCW)".format(rot * 90), log, verb)

                    mirrot_data = mirrot.mirrot(ds["DM"], mir=mir, rot=rot)

                    # If the first map
                    if z_map - params["RedshiftMin"] <= 1e-8:
                        pass
                    else:
                        prev_suffix = "*-{0:03d}.h5".format(i-1)
                        prev_file = utils.glob_files(params["OutputDataDir"],
                                                     params["StackFileName"] +
                                                     prev_suffix)[0]

                        wlog("Combining previous map with current map", log, verb)

                        with h5py.File(prev_file, "r") as prev:
                            prev_data = ds["DM"]
                            mirrot_data += prev_data

                    path = os.path.join(params["OutputDataDir"],
                                        params["StackFileName"])

                    fn = "{0}_z_{1:.5f}-{2:03d}.h5".format(path, z_map, i)
                    with h5py.File(fn, "w") as out:
                        utils.create_dm_dataset(out, mirrot_data)
                        utils.create_redshift_attrs(out, z_map)
                    wlog("Created: {0}".format(fn), log, verb)


                elif params["StackMethod"] == "NoRepeat":
                    # TO DO: WRITE CODE FOR NON REPEATING!
                    pass








    elif not params["PerformStacking"]:
        wlog("Skipping Map Stacking (PerformStacking : False)", log, verb)


    ########################
    # CREATING MASTER DM FILE
    #########################
    wlog("Creating Master DM File", log, verb, t=True)
    if params["CreateMastFile"]:
        stacked_data = utils.glob_files(params["OutputDataDir"], params["StackFileName"] + "*")
        print(stacked_data)
        #unstacked_data = 



        fn = "{0}/{1}.h5".format(params["OutputDataDir"], params["MastFileName"])
        #with h5py.File(fn, "a") as ds:

        print(fn)


    elif not params["CreateMastFile"]:
        wlog("Skipping Master File Creation (CreateMastFile : False)", log, verb)


    


#    Redshift
#    DM_Unstacked
#    DM_Stacked
#    DM_Median
#    DM_Mode
#    DM_Mean
#    DM_Fit

#    wlog("Creating Plots", log, verb, t=True)
#    for i in range(len(projs)):
#        with h5py.File(projs[i], "r") as ds:
#            wlog("Plotting {0}".format(projs[i]), log, verb, u=True)


 #           data1D = utils.reshape_2D_to_1D(ds["DM"], log, verb)

 #           binnum = 100
#            bins = np.linspace(18, 23, binnum)
#
            #f, err = fit.lognormal(data1D, log, verb, boot=False)
#            wlog("Creating DM Histogram", log, verb)
#            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
#            ax = plot.dm_hist(data1D, bins, passed_ax=ax)

            #ax.hist(data1D, density=True, bins=bins)
            #ax.plot(bins, stats.lognorm.pdf(bins, f[0], f[1], f[2]), "r--")
            #ax.set_xlim(bins[0], bins[-1])
            #ax.set_xlabel(r"$\rm{DM\ pc\ cm^{-3}}$", fontsize=18)
            #ax.set_ylabel(r"$\rm{PDF}$")
            #statfit = stats.lognorm.stats(f[0],f[1],f[2])
            #ax.text(0.7, 0.80, r"$\sigma^2 = {0:.3f}$".format(statfit[1]), transform=ax.transAxes, fontsize=16)
            #ax.text(0.7, 0.75, r"$\mu = {0:.3f}$".format(f[0]), transform=ax.transAxes, fontsize=16)
            #ax.text(0.7, 0.70, r"$\mu = {0:.3f}$".format(np.log(f[2])), transform=ax.transAxes, fontsize=16)
 #           plt.tight_layout()
#            wlog("Saving Figure", log, verb)
#            plt.savefig("{0}/TEST_EAGLE_OUT_{1}.png".format(params["PlotDir"], i))
#            plt.close()
#    redshifts_interp = utils.get_redshifts_with_dist_spacing(params["RedshiftMin"],
#                                                                params["RedshiftMax"],
#                                                                params["DistSpacing"])


#    wlog("Performing Mirror and Rotations", log, verb, t=True)

#    dm_bins = np.linspace(0, 1000, 10000)
#
#        #complete_1D_data = np.empty(0)
#        #complete_redshifts = np.empty(0)
#
#    for i in range(len(projs)):
#        print(i, len(projs), len(redshifts_interp))
#        with h5py.File(projs[i], "r") as ds:
#
#            wlog("Plotting {0}".format(projs[i]), log, verb, u=True)
#
#
#            data1D = utils.reshape_2D_to_1D(ds["DM"], log, verb)
#
#            binnum = 100
#            bins = np.linspace(18, 23, binnum)
#
#            #f, err = fit.lognormal(data1D, log, verb, boot=False)
#            wlog("Creating DM Histogram", log, verb)
#            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
#            ax = plot.dm_hist(data1D, bins, passed_ax=ax)
#            plt.tight_layout()
#            wlog("Saving Figure", log, verb)
#            plt.savefig("{0}/TEST_EAGLE_OUT_{1}.png".format(params["PlotDir"], i))
#            plt.close()
#
#
#
#            wlog("MIRROT: {0}".format(projs[i]), log, verb)
#
#            if i == 0:
#
#                with h5py.File("{1}/MIRROT_FILE_{0}.h5".format(i, params["OutputDataDir"]), "w") as mfn:
#                    mfn.create_dataset("DM", data=ds["DM"])
#                    mfn.create_dataset("Redshift", data=redshifts_interp[i])
#
#            else:
#                prev_file = "{1}/MIRROT_FILE_{0}.h5".format(i-1, params["OutputDataDir"])
#
#                with h5py.File(prev_file, "r") as prev:
#                    mir, rot = mirrot_options[random.choice(list(mirrot_options.keys()))]
#
#                    wlog("Mirrot Selection: Mir {0}  Rot: {1} degrees".format(mir, 90*rot), log, verb)
#
#                    mirrot_data = mirrot.mirrot(ds["DM"], mir=mir, rot=rot)
#
#                    with h5py.File("{1}/MIRROT_FILE_{0}.h5".format(i, params["OutputDataDir"]), "w") as mfn:
#                        data = prev["DM"] + mirrot_data
#                        mfn.create_dataset("DM", data=data)
#                        mfn.create_dataset("Redshift", data=redshifts_interp[i])
#                        data1D = utils.reshape_2D_to_1D(data, log, verb)
#                        complete_1D_data = np.append(complete_1D_data, data1D)
#
#
#                        complete_redshifts = np.append(complete_redshifts, [redshifts_interp[i]]*len(data1D))
#                        plot.dm_hist(data1D, bins=dm_bins, fn="{1}/MIRROT_FILE_{0}.png".format(i, params["PlotDir"]))
#
#
#
#
#        mirrot_options = {'r90': (0, 1),
#                          'r180': (0, 2),
#                          'r270': (0, 3),
#                          'm': (1, 0),
#                          'mr90': (1, 1),
#                          'mr180': (1, 2),
#                          'mr270': (1, 3)
#                          }
#    wlog("Creating Master Plot", log, verb, t=True)
#    plot.dmz_2dhist(complete_redshifts, complete_1D_data, bins=(len(redshifts_interp), 100))

    wlog(time.asctime(time.localtime(time.time())), log, verb, t=True)
