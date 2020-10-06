ADMIRE PIPELINE
===============


MODULES
*******
model.py
--------
This is the DMzModel class. It how I load different DM-z relations from
text files or hdf5 files for plotting. This is used mainly in step7 and step8.


transformation.py
-----------------
This module contains all the functions required to perform transformations on the DM
maps such a rotations, mirrors and trnaslations. 

utilities.py
------------
This contains random utilities. Most of these have been replaced by using my utility package `pyx`.


PIPELINE SCRIPTS
****************

pipeline_step01_convert_npz_to_hdf5.py
----------------------------


pipeline_step02_generate_interpolated_maps.py
-----------------------------------


pipeline_step03_generate_interpolated_master.py
-------------------------------------

pipeline_step04_perform_submap_sum.py
---------------------------

pipeline_step05_generate_slice_hist.py
----------------------------

pipeline_step06_sum_sub_map_hist.py
-------------------------

step7_plot_dmz_relation.py
--------------------------

step8_plot_resolution_test.py
-----------------------------
THIS IS A WORK IN PROGRESS



PARAM FILES
***********
These are the parameter files that my pipeline scripts load. Each simulation is loaded
by using a different parameter file.

The current param files I use begin with `pipeline_zinterp`.


SLURM SCRIPTS
*************
These are the slurm scripts that I submit to Ozstar when running my pipeline.


MISC
****
`old_dmz_plot.py` and `open_array.py` were ploting scripts for the DMz relation
however were pretty unusable. So they have been consolidated into `model.py` and
`step7_plot_dmz_relation.py` easy use.



TO DO!
******
- Finish step8

- Make a module called `plots.py` where all the plotting code is located and
  all scripts load that file.

- Change step7 to load a parameter file instead of editing the file and load from plots.

- Rewrite step6 to use a parameter files and much nicer.



