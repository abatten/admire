import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import configparser as CP
import yt
import glob
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
    params["RedshiftMin"] = config.get("Analysis", "RedshiftMin")
    params["RedshiftMax"] = config.get("Analysis", "RedshiftMax")
    params["DistSpacing"] = config.get("Analysis", "DistSpacing")
    params["DistUnits"] = config.get("Analysis", "DistUnits")
    params["LogDir"] = config.get("Log", "LogDir")
    params["LogFileName"] = config.get("Log", "LogFileName" )
    params["YTLogLevel"] = config.getint("Log", "YTLogLevel")
    params["MatplotlibBackend"] = config.get("General", "MatplotlibBackend")

    return params


def create_log_file(log_dir, name):
    """
    Creates the log file for the analysis
    """
    
    fn = os.path.join(log_dir, name)

    # Check if LOG file already exists
    # If so change file name to a unique name
    if os.path.exists(fn):
        index_suffix = 0
        new_fn = fn + "{0:03d}".format(index_suffix)
        while os.path.exists(new_fn):
            index_suffix += 1
            new_fn = fn + "{0:03d}".format(index_suffix)
        fn = new_fn
    log = open(fn, "w")

    # Write header and date/time to the file
    log.write("ADMIRE PIPELINE LOG FILE\n")
    log.write("------------------------\n")
    log.write("{0}\n".format(time.asctime(time.localtime(time.time()))))

    return log


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
    log = create_log_file(params["LogDir"], params["LogFileName"])

    # Write the used parameters to the log file
    log.write("\n{0}\n{1}\n{0}\n".format("----------------", 
                                     "Input Parameters"))

    for key in list(params.keys()):
        log.write("{0:<25} {1}\n".format(key, params[key]))
        if verbose:
           print("{0:<25} {1}".format(key, params[key])) 

    log.write("\n{0}\n".format("-----------------"))

    return params, log


def get_data_for_interpolation(z_sample, redshift_arr, projections,
                               logfile, params):

   # z_snapshots = 
    # The index in the
    logfile.write("Getting Interpolation Data for redshift:{0}\n".format(z_sample))

    idx_low, idx_high = utils.get_idx(z_sample, redshift_arr)
    z_low, z_high = redshift_arr[idx_low], redshift_arr[idx_high]
    dist_low, dist_high = utils.z_to_mpc(z_low), utils.z_to_mpc(z_high)
    proj_low, proj_high = projections[idx_low], projections[idx_high]
    data_low, data_high = h5py.File(proj_low), h5py.File(proj_high)

    logfile.write("{0:<10}{1:<4}{2:<10}{3:<4}\n".format("idx_low", idx_low,
                                               "idx_high", idx_high))

    return data_low, dist_low, data_high, dist_high 

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        params, log = initialise(sys.argv[1], verbose=True)

    else:
        raise OSError("Parameter File not Supplied")

    #mpl.use(params["MatplotlibBackend"])
    yt.mylog.setLevel(params["YTLogLevel"])

    files = glob.glob("{0}/*{1}".format(params["ProjDir"], params["ProjSuffix"]))

    log.write("Reading redshifts of projections\n")
    log.write("-----------------\n")
    if params["Verbose"]:
        print("Reading redshfits of projections")
        print("-----------------")

    snapshot_list = ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"]


    log.write("{0:<25}{1}\n".format("File Name", "Redshift"))
    if params["Verbose"]:
        print("{0:<25}{1}".format("File Name", "Redshift"))
    redshifts = np.empty(0)
    for snap in snapshot_list:
        fn = params["SnapshotDir"] + "/snapshot_0{0}/snap_0{0}.0.hdf5".format(snap)
        snapshot_z = utils.get_redshift_from_snapshot(fn)
        redshifts = np.append(redshifts,snapshot_z)
        log.write("{0:<25}{1}\n".format("snap_0" + snap, snapshot_z))
        if params["Verbose"]:
            print("{0:<25}{1}".format("snap_0" + snap, snapshot_z))
    data_low, dist_low, data_high, dist_high = get_data_for_interpolation(6.7, redshifts, files, log, params)  

# The sample redshift to get an interpolated grid
#sample_z = 6.7

# The index in the redshift array above and below the sample_z
#lower_z_idx, higher_z_idx = utils.get_idx(sample_z, redshifts)

# The redshift of the snapshots above and below the sample_z
#lower_z, higher_z = redshifts[higher_z_idx], redshifts[lower_z_idx]

# The comoving distance of the snapshots above and below the sample_z
#lower_dist, higher_dist = utils.z_to_mpc(lower_z), utils.z_to_mpc(higher_z)

# The snapshots that are above and below in redshift of the sample_z
#lower_snap, higher_snap = files[lower_z_idx], files[higher_z_idx]

# The snapshot_data 
#lower_data, higher_data = h5py.File(lower_snap), h5py.File(higher_snap)

#sample_interp = interpolate.linear_interp_2D(sample_z, lower_data, lower_dist, higher_data, higher_dist)

