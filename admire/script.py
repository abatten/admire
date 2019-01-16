import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import configparser as CP
import yt
import h5py

import utils
import interpolate
import bootstrap
import mirrot

plt.rcParams['text.usetex'] = True


def read_params(param_path, verbose=False):
    """
    Read the supplied parameter file

    Parameters
    ----------
        param_path : string
            The path to the input parameter file.
        verbose : boolean
            Run script verbosely
    """
    params = {}

    params["ParamPath"] = param_path
    params["ParamName"] = param_path.split("/")[-1]
    params["Verbose"] = verbose

    config = CP.ConfigParser()
    config.read(params["ParamPath"])

    params["SnapshotDir"] = config.get("Data", "SnapshotDir")
    params["OutputDataDir"] = config.get("Data", "OutputDataDir")
    params["ProjDir"] = config.get("Data", "ProjDir")
    params["ProjSuffix"] = config.get("Data", "ProjSuffix")
    params["RedshiftMin"] = config.getfloat("Analysis", "RedshiftMin")
    params["RedshiftMax"] = config.getfloat("Analysis", "RedshiftMax")
    params["DistSpacing"] = config.getfloat("Analysis", "DistSpacing")
    params["DistUnits"] = config.get("Analysis", "DistUnits")
    params["LogDir"] = config.get("Log", "LogDir")
    params["LogFileName"] = config.get("Log", "LogFileName" )
    params["YTLogLevel"] = config.getint("Log", "YTLogLevel")
    params["MatplotlibBackend"] = config.get("General", "MatplotlibBackend")

    return params


def initialise(param_path, verbose=False):
    """
    Initialise the pipeline by reading the parameter file and set
    up the log file

    Parameters
    ----------
    param_path : string
        The path to the parameter file for the script

    verbose : boolean
    Run the script verbosely. Default: False

    Returns
    -------
    params : dictionary
        A dictionary with the values from the parameter file.
    log : 
        The log file
    """

    # First read the parameter file
    params = read_params(param_path, verbose)

    # Create the log file for script output
    log = utils.create_log_file(params["LogDir"], params["LogFileName"])

    # Write the used parameters to the log file
    log.write("\n{0}\n{1}\n{0}\n".format("----------------", 
                                     "Input Parameters"))

    for key in list(params.keys()):
        log.write("{0:<25} {1}\n".format(key, params[key]))
        if verbose:
           print("{0:<25} {1}".format(key, params[key])) 

    log.write("\n{0}\n".format("-----------------"))

    return params, log


def get_data_for_interpolation(z_sample, redshift_arr, projections, logfile=None):
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
        logfile.write("\n-----------------")
        logfile.write("\nGetting Interpolation Data: z = {0:.5f}".format(z_sample))
        logfile.write("\n-----------------\n")

    idx_low, idx_high = utils.get_idx(z_sample, redshift_arr)
    z_low, z_high = redshift_arr[idx_low], redshift_arr[idx_high]
    dist_low, dist_high = utils.z_to_mpc(z_low), utils.z_to_mpc(z_high)
    proj_low, proj_high = projections[idx_low], projections[idx_high]
    data_low, data_high = h5py.File(proj_low), h5py.File(proj_high)

    if logfile:
        logfile.write("{0:<10} {1:10}\n{2:<10} {3:10}\n".format("idx_low", idx_low,
                                                                "idx_high", idx_high))
        logfile.write("{0:<10} {1:10.5}\n{2:<10} {3:10.5}\n".format("z_low", z_low, 
                                                                   "z_high", z_high))
        logfile.write("{0:<10} {1:10.5}\n{2:<10} {3:10.5}\n".format("dist_low", dist_low,
                                                                   "dist_high", dist_high))
        logfile.write("{0:<10} {1}\n{2:<10} {3}".format("proj_low", proj_low, 
                                                       "proj_high", proj_high))

    return data_low, dist_low, data_high, dist_high 


def create_interpolated_projections(z_samples, redshifts, files, logfile=None):
    for z in z_samples:
        print("Getting data")
        data_low, dist_low, data_high, dist_high = get_data_for_interpolation(z, redshifts, 
                                                                              files, 
                                                                              logfile=logfile)  

        print("Doing interp")
        interp = interpolate.linear_interp_2D(z, data_low, dist_low, data_high, dist_high, logfile=logfile)



if __name__ == "__main__":

    # The argument to the script is the parameter file
    if len(sys.argv) >= 2:
        params, log = initialise(sys.argv[1], verbose=True)

    else:
        raise OSError("Parameter File not Supplied")

    yt.mylog.setLevel(params["YTLogLevel"])

    files = utils.join_path(params["ProjDir"], "*" + params["ProjSuffix"])

    log.write("Reading redshifts of projections\n")
    log.write("-----------------\n")

    if params["Verbose"]:
        print("Reading redshfits of projections")
        print("-----------------")

    snapshot_list = ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"]


    log.write("{0:<25}{1}\n".format("File Name", "Redshift"))
    if params["Verbose"]:
        print("{0:<25}{1}".format("File Name", "Redshift"))

    snapshot_redshifts = np.empty(0)

    for snap in snapshot_list:
        fn = params["SnapshotDir"] + "/snapshot_0{0}/snap_0{0}.0.hdf5".format(snap)
        snapshot_z = utils.get_redshift_from_snapshot(fn)
        snapshot_redshifts = np.append(snapshot_redshifts,snapshot_z)
        log.write("{0:<25}{1}\n".format("snap_0" + snap, snapshot_z))
        if params["Verbose"]:
            print("{0:<25}{1}".format("snap_0" + snap, snapshot_z))

    if params["Verbose"]:
        print("Calculating the redshifts needed for interpolation")

    # Calculate the list of redshifts to interpolate from the Min/Max Redshift
    # and the DistSpacing
    redshifts_to_interp = utils.get_redshifts_with_dist_spacing(params["RedshiftMin"], 
                                                                  params["RedshiftMax"], 
                                                                  params["DistSpacing"],
                                                                  logfile=log)
    if params["Verbose"]:
        print("List of Redshifts: {0}".format(redshifts_to_interp))

    # Create the interpolated projections, then plot and save them to .h5 files
    create_interpolated_projections(redshifts_to_interp, snapshot_redshifts, files, logfile=log)

#    for i in range(len(interp_proj_redshifts)):
#        print("Getting data")
#        data_low, dist_low, data_high, dist_high = get_data_for_interpolation(interp_proj_redshifts[i], redshifts, files, log)  

#        print("Doing interp")
#        interp = interpolate.linear_interp_2D(interp_proj_redshifts[i], data_low, dist_low, data_high, dist_high, logfile=log)

