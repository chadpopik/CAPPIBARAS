# Cross-spectra and Average Profile Predictions for Inference of Baryonic Astrophysics off high-Resolution Astronomical Surveys (CAPPIBARAS)


Still in development. Status of modules:
- HODs, SHMRs, Dust only depend on what's in the .py file and should work easily. 
- HaloModels and FFTs require some additional packages that needed to be installed but should work easily otherwise, and Profiles requires the use of these two but should work fine if they do. 
- SMFs and Data require files that are on NERSC or aren't necessarily public, so some things in these might not be functional without those files.
- Spectra and Projections are still being checked for accuracy and under development.
- Any part of running chains like SOLikelihoods/runchains need to be revisited and updated and are likely not functional right now.

First run ModuleCheck.ipynb, which will explain and run every one of the individual model modules and ensure they work, then ForwardModelCheck.ipynb, which compiles the entire pipeline from profile to signal. 
