"""
This module contains the ``DMzModel`` class. The ``DMzModel class handles all the
data and paths for loading DM-z relations.

"""

import numpy as np

class DMzModel:
    """
    Handles all of the data and paths for loading DM-z relations.


    txt files:
        Redshift and DM values are accessed through model.z_vals and model.DM_vals

    hdf5 files:
        Redshift bins, DM_bins and 2D Histogram are accessed through model.z_bins
        model.DM_bins, model.Hist
    """
    def __init__(self, model_dict):
        """

        Parameters
        ----------
        model_dict: dict [string, variable]
            Dictionary containing parameter values for this class instance.
        """

        self.model_dict = model_dict

        acceptable_model_categories = set({
            "1D-analytic",
            "1D-semi-analytic",
            "1D-hydrodynamic",
            "2D-hydrodynamic",
            "mean-hydrodynamic",
            "std-hydrodynamic",
        })

        if model_dict["category"] not in acceptable_model_categories:
            msg = ("Invalid 'model_type' entered. {0} is not in the list of "
                   "available model types. Allowed types are: "
                   "{1}".format(model_dict["category"],
                                acceptable_model_categories))
            raise ValueError(msg)


        # Set the attributes we were passed.
        for key in model_dict:
            setattr(self, key, model_dict[key])

        self.load_model()


    def load_model(self):
        """
        Loads the DM-z Model.


        Checks which file type the model is (txt or hdf5) and loads all the data.
        """

        if self.model_dict["category"][:2] == "1D":
            if self.file_format in set(["txt", "TXT", ".txt", ".TXT"]):
                self.z_vals, self.DM_vals = np.genfromtxt(self.path, unpack=True)

        elif self.model_dict["category"][:4] == "mean":
                data = np.loadtxt(self.path, unpack=True, skiprows=2)
                self.z_vals = data[0]
                self.DM_vals = data[1]

        elif self.model_dict["category"][:3] == "std":
                data = np.loadtxt(self.path, unpack=True, skiprows=2)
                self.z_vals = data[0]
                self.sigma1_low = data[3]
                self.sigma1_upp = data[4]
                self.sigma2_low = data[5]
                self.sigma2_upp = data[6]
                self.sigma3_low = data[7]
                self.sigma3_upp = data[8]

        elif self.file_format in set(["hdf5", "HDF5", "h5", ".hdf5", ".HDF5", ".h5"]):
            import h5py

            with h5py.File(self.path, "r") as ds:
                self.z_bins = ds["Redshifts_Bin_Edges"][1:]
                self.DM_bins = ds["DM_Bin_Edges"][:]
                self.DM_bin_widths = ds["DM_Bin_Widths"][:]
                self.Hist = ds["DMz_hist"][:]
                self.DM_bin_centres = ds["DM_Bin_Centres"][:]

        else:
            msg = ("I don't know how to load a {} "
                   "file format".format(self.file_format))
            raise ValueError(msg)
