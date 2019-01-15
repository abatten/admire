import utils
import bootstrap
import mirrot
import numpy as np
import yt
import glob
import h5py
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import interpolate
import time
import configparser as CP
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
        index_suffix = 1
        while os.path.exists(fn+"{0}".format(index_suffix)):
            index_suffix += 1
        fn = fn + "{0}".format(index_suffix)

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
        log.write("{0:<20} {1}\n".format(key, params[key]))
        if verbose:
           print("{0:<20} {1}".format(key, params[key])) 

    log.write("\n{0}\n".format("-----------------"))

    return params, log


def get_data_for_interpolation(z_sample):

   # z_snapshots = 
    z_low_idx, z_high_idx = utils.get_idx(z_sample, z_snapshots)


if __name__ == "__main__":

    if len(sys.argv) >= 2:
        params, log = initialise(sys.argv[1], verbose=True)

    else:
        raise OSError("Parameter File not Supplied")

    mpl.use(params["MatplotlibBackend"])
    yt.mylog.setLevel(params["YTLogLevel"])

    z = utils.get_redshift_from_snapshot("/Users/abatten/PhD/data/AURORA/L012N0128/Aurora_L012N0128_FSN1.0_FESC0.5/data/snapshot_050/snap_050.0.hdf5")
    print(z)

    files = glob.glob("{0}/*{1}".format(params["ProjDir"], params["ProjSuffix"]))
    print(files)


    log.write("Reading redshifts of projections\n")
    if params["Verbose"]:
        print("Reading redshfits of projections\n")

    snapshot_list = ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"]

    redshifts = np.empty(0)
    for snap in snapshot_list:
        fn = params["SnapshotDir"] + "/snapshot_0{0}/snap_0{0}.0.hdf5".format(snap)
        redshifts = np.append(redshifts, utils.get_redshift_from_snapshot(fn))
    print(redshifts)

# The sample redshift to get an interpolated grid
sample_z = 6.7

# The index in the redshift array above and below the sample_z
lower_z_idx, higher_z_idx = utils.get_idx(sample_z, redshifts)

# The redshift of the snapshots above and below the sample_z
lower_z, higher_z = redshifts[higher_z_idx], redshifts[lower_z_idx]

# The comoving distance of the snapshots above and below the sample_z
lower_dist, higher_dist = utils.z_to_mpc(lower_z), utils.z_to_mpc(higher_z)

# The snapshots that are above and below in redshift of the sample_z
lower_snap, higher_snap = files[lower_z_idx], files[higher_z_idx]

# The snapshot_data 
lower_data, higher_data = h5py.File(lower_snap), h5py.File(higher_snap)

sample_interp = interpolate.linear_interp_2D(sample_z, lower_data, lower_dist, higher_data, higher_dist)

