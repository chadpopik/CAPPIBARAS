Cross-spectra and Average Profile Predictions for Inference of Baryonic Astrophysics off high-Resolution Astronomical Surveys (CAPPIBARAS)

Current Structure:

Basic structure is this:

- README.md : This file

- config.py : Collection of common and easy to install packages to use throughout CAPPIBARAS

- config_local.py (WRITE AND ADD TO .gitignore): Contains file locations on whatever system you're using CAPPIBARAS on to point the code towards
  - DATA_PATH, location of any and all data used in the forward model (catalogs, measurements, target distributions, etc)
  - SOLIKET_PATH, location of the SOLikeT package
  - OUTPUT_PATH, location to put the output of cobaya jobs

- Models : contains various components that are used in forward model, and can be used separately
  - Papers: Collection of .py files that contain all the models (and relevant figures/tables) from publications
    - Paper1.py, Paper2.py, ..., PaperN.py
  - Checks: Collection of .ipynb files used to check the outputs of the models and compare against plots in the paper
    - Paper1.ipynb, Paper2.ipynb, ..., PaperN.ipynb
  - Figures: Collection of folders that contain tables with data to use in the models, and figures to use to checks the results of the models
    - Paper1/, Paper2/, ..., PaperN/

- FowardModelTEST.ipynb (IN PROGRESS): Notebook to check various parts of the forward model

- yamls : Collection of yaml files used to run cobaya
  - yaml1.py, yaml2.py, ..., yamlN.py

- checkchains.ipynb : Notebook to check SOLikeT implementation of Forward Model and results from MCMC fitting 

- SOLikelihoods.py : Joins the forward model with SOLikeT for use with cobaya

- runchains.py : File for running cobaya fitting jobs


- runchains_local/NERSC/rusty.sh : batch files for submitting a job to local node or to a cluster as a slurm

  - submit_chain.py (IN PROGRESS) :  python file to create the .sh job to submit instead of having to have multiple .sh jobs
