# Cross-spectra and Average Profile Predictions for Inference of Baryonic Astrophysics off high-Resolution Astronomical Surveys (CAPPIBARAS)

Current Structure:

Basic structure is this:
- Models: contains various components that are used in forward model, and can be used separately
    - For each type of submodel, there is also a .ipynb notebook used to check to see if the submodels are working




- README.md : This file
- config.py : Collection of common and easy to install packages to use throughout CAPPIBARAS
  - note that you will need to make a config_local.py file at the same level as this file that define the following:
    - DATA_PATH, location of any and all data used in the forward model (catalogs, measurements, target distributions, etc)
    - SOLIKET_PATH, location of the SOLikeT package
    - OUTPUT_PATH, location to put the output of cobaya jobs
- FowardModelCheck.ipynb (likely depreciated): Notebook to check various parts of the forward model
- checkchains


- SOLikelihoods.py : Joins the forward model with SOLikeT for use with cobaya 
- runchains.py : File for running cobaya fitting jobs
- runchains_local/NERSC/rusty.sh : batch files for submitting a job to local node or to a cluster as a slurm
    - submit_chain.py : (In progress to replace .sh files) python file to create the .sh job to submit instead of having to have multiple .sh jobs 
-  yamls