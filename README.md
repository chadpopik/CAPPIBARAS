# Capybara (placeholder name)


Very initial version of the code, no judging.

Things to come:
- Mass definition conversions
- DESI SMF from catalog
- twohalo theory term (currently just using one from a file)
- Calculating C_ells 
- Miscentering (maybe)



Profiles:
- TODO 1: Add a two-halo term that's constructed from theory?


HODs:
- TODO 1: Add valid mass and redshift ranges for each model
- TODO 2: Check to see if the other HODs have extra parts of their models, like different satellite profiles


SHMRs:
- TODO 1: Add valid mass and redshift ranges for each model
- TODO 3: Add satellite dependent model somehow


HMFs:
- TODO 1: Investigate more the cosmology dependence for both models


SMFs:
- TODO 1: Check for units and factors of h/dex
- TODO 2: Add zdistributions for CMASS


Measurements:
- TODO 1: Find better source of Schaan measurements/covariances and dust model (just using Emily's TNG for now). Also why is the covariances off from the error, shouldn't they be the same?
- TODO 2: What are the units on the beam response and how should it be converted? Using it raw isn't right.
- TODO 3: Find Liu response, and probably have to figure out conversions same as TODO 2.
- TODO 4: Find dust model to use for Liu, just using Emily's TNG file for now.
- TODO 5: Ask about dust file, should it be negative?
- TODO 6: Find beam and responses for boryana measurements


FFTs:
- TODO 1: test mcfit for forward/backward equivalency
- TODO 2: understand better Emily's Hankel transform


ForwardModel:
- TODO 1: clean up the projection and aperture functions
- TODO 2: Maybe too much of th FFT mess is done in this file, it can be put into the FFT file?
- TODO 3: check if intergrating over zs is really necessary, maybe just use the medium z?


SOLikelihoods:
- TODO 1: Is there a way to not way to write a preinit line for each value that's needed?
- TODO 2: Is there a way to just know the yaml file and more easily load the original params?
- TODO 3: Should I explore with actually creating a theory class to calculate some of the HOD/HMF type things?
- TODO 4: How to decide rs array? 


runchains:
- TODO 1: Make the corner plot automatically find and assign the fit parameters (.py)
- TODO 2: Find a way to make the general model things (HOD, SHMR, mass_function, etc) apply to every likelihood