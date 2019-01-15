import utils
import bootstrap
import mirrot
import numpy as np
import yt
import glob
import h5py
import os
import matplotlib.pyplot as plt
import interpolate

import configparser as CP

def read_params(param_path, verbose=False):
    """
    Read the supplied parameter file

    Parameters
    ----------
        param_path : string
            The path to the input parameter file.
        verbose : bool
            Run script verbosely
    """
    params = {}

    params["param_path"] = param_path
    params["param_name"] = param_path.split("/")[-1]
    params["verbose"] = verbose

    config = CP.ConfigParser()
    config.read(params["params_name"])

    params["SnapshotDir"] = config.get("General", "SnapshotDir")
    params["OutputDataDir"] = config.get("General", "OutputDataDir")
    params["LogFileName"] = config.get("LOG", "LogFileName" )
    params["YTLogLevel"] = config.getint("LOG", "YTLogLevel")
    params["MatplotlibBackend"] = config.get("General", "MatplotlibBackend")

    return params


def create_log_file(file_name, output_dir):

    loc = os.path.join(output_dir, file_name)

    if os.path.exists(loc):
        index_suffix = 0
        while os.path.exists(loc+"{0}".format(index_suffix)):
            index_suffix += 1
        loc = loc + "{0}".format(index_suffix)

    


    return loc


print(create_log_file("ADMIRE.LOG", "/Users/abatten/PhD/admire/admire"))

yt.mylog.setLevel(50)
plt.rcParams['text.usetex'] = True

data_dir = "/Users/abatten/PhD/data/AURORA/L012N0128/Aurora_L012N0128_FSN1.0_FESC0.5/data"
files = glob.glob("/Users/abatten/PhD/borealis/notebooks/*_proj.h5")
snapshot_list = ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"]

redshifts = np.empty(0)
for snap in snapshot_list:
    fn = data_dir + "/snapshot_0{0}/snap_0{0}.0.hdf5".format(snap)
    ds = yt.load(fn)
    redshifts = np.append(redshifts, ds["Redshift"])

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

sample_interp = interpolate.linear_interp_2D(sample_z, lower_data, lower_dist, 
                    higher_data, higher_dist)

