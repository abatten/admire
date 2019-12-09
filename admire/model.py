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



        acceptable_model_categories = [
            "1D-analytic",
            "1D-semi-analytic",
            "1D-hydrodynamic",
            "2D-hydrodynamic",
        ]

        if model_dict["category"] not in acceptable_model_categories:
            msg = ("Invalid 'model_type' entered. {0} is not in the list of "
                   "available model types. Allowed types are: "
                   "{1}".format(model_dict["category"],
                                acceptable_model_types))
            raise ValueError(msg)


        # Set the attributes we were passed.
        for key in model_dict:
            setattr(self, key, model_dict[key])

        self.load_model()



    
    def load_model(self):
        """
        Loads the DM-z Model.


        CHecks which file type the model is (txt or hdf5) and loads all the data.
        """
        if self.file_format in ["txt", "TXT", ".txt", ".TXT"]:
            self.z_vals, self.DM_vals = np.genfromtxt(self.path, unpack=True)

        elif self.file_format in ["hdf5", "HDF5", "h5", ".hdf5", ".HDF5", ".h5"]:
            import h5py
            
            with h5py.File(self.path, "r") as ds:
                self.z_bins = ds["Redshifts"][:]
                self.DM_bins = ds["Bin_Edges"][:]
                self.Hist = ds["DMz_hist"][:]



        else:
            msg = ("I don't know how to load a {} " 
                   "file format".format(self.file_format))  
            raise ValueError(msg)



#    def calc_DM_bin_centres(self):
