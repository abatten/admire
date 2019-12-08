import os
import numpy as np
from model import DMzModel
    



def calc_mean_from_hist(hist, z_bins, dm_bins):



    for idx,  z in enumerate(z_bins):
        z_mean = np.sum(hist[:, idx] * dm_bins)
        print(z_mean)
    
   
 



def calc_std_from_hist(hist, z_bins, dm_bins):


def calc_mode_from_hist(hist, z_bins, dm_bins):









if __name__ == "__main__":    
    
    
    RefL0025N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0752/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed.hdf5",
        "label"        : "RefL0025N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.rainforest_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }
    
    
    
    RecalL0025N0752 = {
        "dir_name"     : "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RecalL0025N0752/all_snapshot_data/output/T4EOS",
        "file_name"    : "admire_output_DM_z_hist_total_normed.hdf5",
        "label"        : "RecalL0025N0752",
        "file_format"  : "hdf5",
        "category"     : "2D-hydrodynamic",
        "dm_scale"     : "linear",
        "color"        : cmasher.rainforest_r,
        "linestyle"    : None,
        "linewidth"    : None,
        "marker"       : None,
        "plot_toggle"  : True,
    }
    
    model_dicts = [
        RefL0025N0752,
        RecalL0025N0752
    ]

#######################################################################
#######################################################################
#                         DO NOT TOUCH
#######################################################################
#######################################################################

    for model in model_dicts:
        path = os.path.join(model["dir_name"], model["file_name"])
        model["path"] = path

    all_models = []
    for model_dict in model_dicts:
        model = DMzModel(model_dict)
        all_models.append(model)
    

    calc_mean_from_hist()

    #plot_dmz_relation(all_models, "dmz_relation_full_RecalL0025N0752", "")

    print_tools.print_footer()


