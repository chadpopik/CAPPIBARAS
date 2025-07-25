# Capybara (placeholder name)


Very initial version of the code, no judging.

Things to come:
- Mass definition conversions
- DESI SMF from catalog
- Calculating C_ells
- Miscentering (maybe)
- More details on mass/redshift ranges for all the models
- Satellite dependent SHMR model



HODs:
- TODO 1: Check to see if the other HODs have extra parts of their models, like different satellite profiles, and add if necessary


SHMRs:
- 


HMFs:
- TODO 1: Investigate more detail to the cosmology dependence for both models


SMFs:
- TODO 1: Check for units and factors of h/dex
- TODO 2: Add zdistributions for CMASS


FFTs:
- TODO 1: test mcfit for forward/backward equivalency
- TODO 2: understand better Emily's Hankel transform


Profiles:
- TODO 1: Various two-halo checks and tasks: how to handle the FFT? whether to take in functions or array for Plin? Specify the model? Check for mass definitions?


Measurements:
- TODO 1: Find better source of Schaan measurements/covariances and dust model (covariances off from the error?), and dust profile
- TODO 2: Find beam and response for Henry and Boryana measurements
- TODO 3: Find dust profile for Henry and Boryana measurements


ForwardModel:
- TODO 1: clean up the projection and aperture functions
- TODO 2: Maybe too much of th FFT mess is done in this file, it can be put into the FFT file?
- TODO 3: check if intergrating over zs is really necessary, maybe just use the medium z?
- TODO 4: Should i just drop the y conversion? like why is it in there? for rho it's incorrect, and for Pth it might be unneeded.



SOLikelihoods:
- TODO 1: Is there a way to not way to write a preinit line for each value that's needed?
- TODO 2: Is there a way to just know the yaml file and more easily load the original params?
- TODO 3: Should I explore with actually creating a theory class to calculate some of the HOD/HMF type things?
- TODO 4: How to decide rs array? 


runchains:
- TODO 1: Find a way to make the general model things (HOD, SHMR, mass_function, etc) apply to every likelihood