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
    params["ProjDir"] = config.get("Data", "ProjDir")
    params["ColDensMapSuffix"] = config.get("Data", "ColDensMapSuffix")
    params["ColDensMapZVals"] = json.loads(config.get("Data", "ColDensMapZVals"))

    params["InterpMaps"] = config.getboolean("Interpolate", "InterpMaps")
    params["InterpFileName"] = config.get("Interpolate", "InterpFileName")

    params["RedshiftMin"] = config.getfloat("Analysis", "RedshiftMin")
    params["RedshiftMax"] = config.getfloat("Analysis", "RedshiftMax")
    params["DistSpacing"] = config.getfloat("Analysis", "DistSpacing")
    params["DistUnits"] = config.get("Analysis", "DistUnits")

    params["PlotDir"] = config.get("Plot", "PlotDir")
    params["CreateColDensMap"] : config.getboolean("Plot", "CreateColsDensMap")
    params["ColDensMapFileName"] = config.get("Plot", "ColDensMapFileName")
    params["ColDensCmap"] = config.get("Plot", "ColDensCmap")
    params["ColDensVmax"] = config.getfloat("Plot", "ColDensVmax")
    params["ColDensVmin"] = config.getfloat("Plot", "ColDensVmin")
    params["CreateDMPdf"] = config.getboolean("Plot", "CreateDMPdf")
    params["DMPdfFileName"] = config.get("Plot", "DMPdfFileName")
    params["CreateMastHist"] = config.getboolean("Plot", "CreateMastHist")
    params["MastHistFileName"] = config.get("Plot", "MastHistFileName")
    params["MatplotlibBackend"] = config.get("Plot", "MatplotlibBackend")

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


def get_data_for_interpolation(z_sample, redshift_arr, projections, logfile=None, verbose=True):
    """
    Finds the two projections with redshifts that are the nearest higher and
    nearest lower redshifts and extracts the

    Parameters
    ----------
    z_sample : float
        The redshift of interest in the interpolation.

    redshift_arr: array or array-like
        The redshifts from the projections ordered by snapshot number.

    projections : array or array-like
        The filenames of the projections. These have the same indexing as
        redshift_arr

    logfile :
        The file to write the logs.


    Returns
    -------
    data_low :
        The data of the projection with the nearest lower redshift to z_sample.

    dist_low :
        The comoving distance to the projection with the nearest lower
        redshift to z_sample.

    data_high :
        The data of the projection with the nearest higher redshift to z_sample.

    dist_high :
        The comoving distance to the projection with the nearest higher
        redshift to z_sample.
    """

    if logfile:
        wlog("Getting Interpolation Data: z = {0:.5f}".format(z_sample), logfile, verbose, t=True)

    idx_low, idx_high = utils.get_idx(z_sample, redshift_arr)
    z_low, z_high = redshift_arr[idx_low], redshift_arr[idx_high]
    dist_low, dist_high = utils.z_to_mpc(z_low), utils.z_to_mpc(z_high)
    proj_low, proj_high = projections[idx_low], projections[idx_high]
    data_low, data_high = h5py.File(proj_low), h5py.File(proj_high)

    if logfile:
        wlog("{0:<10} {1:10}\n{2:<10} {3:10}\n".format("idx_low", idx_low,"idx_high", idx_high), logfile, verbose)
        wlog("{0:<10} {1:10.5}\n{2:<10} {3:10.5}\n".format("z_low", z_low, "z_high", z_high), logfile, verbose)
        wlog("{0:<10} {1:10.5}\n{2:<10} {3:10.5}\n".format("dist_low", dist_low,"dist_high", dist_high), logfile, verbose)
        wlog("{0:<10} {1}\n{2:<10} {3}".format("proj_low", proj_low, "proj_high", proj_high), logfile, verbose)

    return data_low, dist_low, data_high, dist_high


def create_interpolated_projections(z_samples, redshifts, files, params, logfile=None, verbose=False):
    for z in z_samples:
        (data_low, dist_low, 
         data_high, dist_high) = get_data_for_interpolation(z, redshifts, files, logfile=logfile)  

        interp = interpolate.linear_interp_2D(z, data_low, dist_low, 
                                              data_high, dist_high, 
                                              logfile=logfile)

        output_file = os.path.join(params["OutputDataDir"], 
                                   "{0}_z_{1:.3f}.h5".format(params["InterpFileName"], z))

        plot.projection(interp, z, params, logfile=log)

        with h5py.File(output_file, "w") as h5f:
            if logfile:
                wlog("Creating Interpolated Slice: {0}".format(output_file), logfile, verbose)
            h5f.create_dataset("DM", data=interp.value)


if __name__ == "__main__":

    # The argument to the script is the parameter file
    if len(sys.argv) >= 2:
        if os.path.exists(sys.argv[1]):
            params, log = initialise(sys.argv[1])
        else:
            raise OSError("Could not find file: {0}".format(sys.argv[1]))

    else:
        raise OSError("Parameter File not Supplied")

    # Set the verbose level of the script
    yt.mylog.setLevel(params["YTLogLevel"])
    verb = params["Verbose"]

    wlog("Reading Column Density Map Data", log, verb, t=True)

    npz_format = [".npz", "npz"]
    h5_format = [".h5", "h5"]

    # Need to speficy the redshifts of the npz files
    ColDensMapsZVals = params["ColDensMapZVals"]

    # If the ColDens Maps are npz files convert them to hdf5 files
    if params["ColDensMapSuffix"] in npz_format:
        wlog("Column Density Maps in .npz format", log, verb)
        wlog("Converting files to .h5 files", log, verb)
        ColDensMaps = utils.glob_files(params["ColDensMapDir"],
                                        "*" + params["ColDensMapSuffix"])

        for i in range(len(ColDensMaps)):
            old_fn = ColDensMaps[i].split(".npz")[0].split("/")[-1]
            new_fn = "{0}/{1}.h5".format(params["OutputDataDir"], old_fn)
            redshift = np.array(ColDensMapsZVals[i])

            wlog("Converting {0} to .h5 format".format(old_fn), log, verb)
            utils.convert_npz_to_h5(ColDensMaps[i], new_fn, z=redshift)
            wlog("Created: {0}".format(new_fn), log, verb)

            # Change file path to the path of the new .h5 file
            ColDensMaps[i] = new_fn

    elif ColDensMapSuffix in h5_format:
        wlog("Column Density Maps in .h5 format", log, verb)
        ColDensMaps = utils.glob_files(params["ColDensMapDir"],
                                       "*" + params["ColDensMapSuffix"])

    else:
        raise TypeError("Unrecognised format type. ColDensMaps should be in\
                        .npz or .h5 format.")
        sys.exit(1)


    # Read the redshifts 
    wlog("Reading redshifts of projections", log, verb, u=True)
    wlog("{0:<10} {1:<50}".format("Redshift", "Filename"), log, verb)
    for m in ColDensMaps:
        with h5py.File(m, "r") as ds:
            z = ds["Header"].attrs["Redshift"]
            fn = m.split("/")[-1]
            wlog("{0:<10.5f} {1:<50}".format(z, fn), log, verb)

#    # Get the redshifts of the snapshots
#    snapshot_redshifts = np.empty(0)
#    for snap in snapshot_list:
#        fn = params["ColDensMapDir"] + "/snapshot_0{0}/snap_0{0}.0.hdf5".format(snap)
#        snapshot_z = utils.get_redshift_from_snapshot(fn)
#        snapshot_redshifts = np.append(snapshot_redshifts,snapshot_z)

#        log.write("{0:<25}{1}\n".format("snap_0" + snap, snapshot_z))
#        if params["Verbose"]:
#            print("{0:<25}{1}".format("snap_0" + snap, snapshot_z))

    if params["InterpMaps"]:
        wlog("Calculating the redshifts needed for interpolation", log, verb, u=True)

        # Calculate the list of redshifts to interpolate from the Min/Max Redshift
        # and the DistSpacing
        redshifts_to_interp = utils.get_redshifts_with_dist_spacing(params["RedshiftMin"],
                                                                    params["RedshiftMax"],
                                                                    params["DistSpacing"])
        wlog("List of Redshifts:\n{0}".format(redshifts_to_interp), log, verb)

        # Create the interpolated projections, then plot and save them to .h5 files
        create_interpolated_projections(redshifts_to_interp,
                                        ColDensMapZVals,
                                        files, params, logfile=log,
                                        verbose=verb)

        projs = utils.glob_files(params["OutputDataDir"], params["InterpFileName"] + "*")

    else:
        projs = utils.glob_files(params["ProjDir"],  params["InterpFileName"] + "*")


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
    redshifts_to_interp = utils.get_redshifts_with_dist_spacing(params["RedshiftMin"],
                                                                params["RedshiftMax"],
                                                                params["DistSpacing"])


    wlog("Performing Mirror and Rotations", log, verb, t=True)

    dm_bins = np.linspace(0, 1000, 10000)

    complete_1D_data = np.empty(0)
    complete_redshifts = np.empty(0)

    for i in range(len(projs)):
        print(i, len(projs), len(redshifts_to_interp))
        with h5py.File(projs[i], "r") as ds:

            wlog("Plotting {0}".format(projs[i]), log, verb, u=True)


            data1D = utils.reshape_2D_to_1D(ds["DM"], log, verb)

            binnum = 100
            bins = np.linspace(18, 23, binnum)

            #f, err = fit.lognormal(data1D, log, verb, boot=False)
            wlog("Creating DM Histogram", log, verb)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
            ax = plot.dm_hist(data1D, bins, passed_ax=ax)
            plt.tight_layout()
            wlog("Saving Figure", log, verb)
            plt.savefig("{0}/TEST_EAGLE_OUT_{1}.png".format(params["PlotDir"], i))
            plt.close()



            wlog("MIRROT: {0}".format(projs[i]), log, verb)

            if i == 0:

                with h5py.File("{1}/MIRROT_FILE_{0}.h5".format(i, params["OutputDataDir"]), "w") as mfn:
                    mfn.create_dataset("DM", data=ds["DM"])
                    mfn.create_dataset("Redshift", data=redshifts_to_interp[i])

            else:
                prev_file = "{1}/MIRROT_FILE_{0}.h5".format(i-1, params["OutputDataDir"])

                with h5py.File(prev_file, "r") as prev:
                    mir, rot = mirrot_options[random.choice(list(mirrot_options.keys()))]

                    wlog("Mirrot Selection: Mir {0}  Rot: {1} degrees".format(mir, 90*rot), log, verb)

                    mirrot_data = mirrot.mirrot(ds["DM"], mir=mir, rot=rot)

                    with h5py.File("{1}/MIRROT_FILE_{0}.h5".format(i, params["OutputDataDir"]), "w") as mfn:
                        data = prev["DM"] + mirrot_data
                        mfn.create_dataset("DM", data=data)
                        mfn.create_dataset("Redshift", data=redshifts_to_interp[i])
                        data1D = utils.reshape_2D_to_1D(data, log, verb)
                        complete_1D_data = np.append(complete_1D_data, data1D)


                        complete_redshifts = np.append(complete_redshifts, [redshifts_to_interp[i]]*len(data1D))
                        plot.dm_hist(data1D, bins=dm_bins, fn="{1}/MIRROT_FILE_{0}.png".format(i, params["PlotDir"]))




        mirrot_options = {'r90': (0, 1),
                          'r180': (0, 2),
                          'r270': (0, 3),
                          'm': (1, 0),
                          'mr90': (1, 1),
                          'mr180': (1, 2),
                          'mr270': (1, 3)
                          }
    wlog("Creating Master Plot", log, verb, t=True)
    plot.dmz_2dhist(complete_redshifts, complete_1D_data, bins=(len(redshifts_to_interp), 100))

    wlog(time.asctime(time.localtime(time.time())), log, verb, t=True)
