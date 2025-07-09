# Capybara (placeholder name)

Very initial version of the code



HODs:
- TODO 1: Check to see if I should be adding sattelite profiles for the rest of the HODs


SHMRs:
- TODO 1: Figure out how to convert from one mass definition to another, probably using packages that we use to get the HMFs
- TODO 2: Add valid mass ranges


HMFs:
- TODO 1: Check hmf against pyccl
- TODO 2: Check for units of input M and units of output SMF (watch for factors of h/dex)
- TODO 3: Consider interpolating hmf to actually match the input ms rather then use them for range guidance?
- TODO 4: Add cosmology dependence for both models


SMFs:
- TODO 1: Check for units and factors of h/dex
- TODO 2: Clean CMASS DR10 and making adaptable to ms and zs
- TODO 3: Make the class more organized and sensible with functionality


Profiles:
- TODO 1: Add a two-halo term that's constructed from theory?


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
- TODO 4: Add miscentering
- TODO 5: Add calculation for Cls


SOLikelihoods:
- TODO 1: Is there a way to not way to write a preinit line for each value that's needed?
- TODO 2: Is there a way to just know the yaml file and more easily load the original params?
- TODO 3: Should I explore with actually creating a theory class to calculate some of the HOD/HMF type things?
- TODO 4: How to decide rs array? 


runchains:
- TODO 1: Make the corner plot automatically find and assign the fit parameters (.py)
- TODO 2: Find a way to make the general model things (HOD, SHMR, mass_function, etc) apply to every likelihood