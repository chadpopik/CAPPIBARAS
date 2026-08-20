Cross-spectra and Average Profile Predictions for Inference of Baryonic Astrophysics off high-Resolution Astronomical Surveys (CAPPIBARAS)

1. Check your config files:
- config.py : Collection of common and easy to install packages to use throughout CAPPIBARAS
  - config_local.py (WRITE AND ADD TO .gitignore): Contains file locations on whatever system you're using CAPPIBARAS on to point the code towards

2. Then test your individual models that you want to use:
- Models : contains various components that are used in forward model, and can be used separately
  - Codes:Collection of .py files that contain packages imported from outside
  - Papers: Collection of .py files that contain models (and relevant figures/tables) from publications
    - Paper1.py, Paper2.py, ..., PaperN.py
    - Checks: Collection of .ipynb files used to check the outputs of the models and compare against plots in the paper
      - Paper1.ipynb, Paper2.ipynb, ..., PaperN.ipynb
    - Figures: Collection of folders that contain tables with data to use in the models, and figures to use to checks the results of the models
      - Paper1/, Paper2/, ..., PaperN/
  - Dust.py, FFTs.py, HaloModels.py, HODs.py, etc. anything else on this level is depriciated, don't use it

3. Then you can test the model in FowardModel.ipynb, which calls on ForwardModel.py which uses aspects of the Models:

4. Then you check to see if it's loading in and working properly in Cobaya in in checkchains.py, which uses the code from SOLikelihoods.py and one of the yaml files from the yamls folder.

5. Then you can run the file using runchains.py, or run that file on a cluster using on of the .sh files

submit_chain.py (IN PROGRESS) :  python file to create the .sh job to submit instead of having to have multiple .sh jobs
