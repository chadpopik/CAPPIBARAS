"""
Collection of Halo Mass Function obtained either from halo model codes or loaded from a data files. 
Classes should contain halo number density 2D arrays over halo mass (in m200c) and redshift and the corresponding mass/redshift arrays, in consistent units. If using functions, require input halo mass/redshift arrays to ensure output consistency between classes.


TODO 1: Check hmf against others
TODO 2: Check for units of input M and units of output SMF (watch for factors of h/dex)
TODO 3: Consider interpolating hmf to actually match the input ms rather then use them for range guidance?
"""

from Basics import *


class BASEHMF:
    def checkmodel(self, model):
        if model in self.models:  # Check if the model is in the list of models
            self.model = model
        else: 
            raise NameError(f"Model {model} doesn't exist, choose from available samples: {self.models}")


class hmf_package(BASEHMF):  # https://hmf.readthaedocs.io/en/latest/index.html
    def __init__(self, model):
        # Import package only if using this model
        import hmf
        self.hmf = hmf
        
        self.models = dir(hmf.mass_function.fitting_functions)[0:30]
        self.checkmodel(model)

    def HMF(self, ms, zs):
        # Set spacing between logMhalo values, which must be logspaced
        try: dlog10m = np.min(np.log10(ms[1:])-np.log10(ms[:-1]))  # TODO 3
        except ValueError: dlog10m = 0.1

        # Function only takes one z at a time so use list comprehension and then combine
        haloMFsraw = [self.hmf.MassFunction(z=z, Mmin=np.log10(np.min(ms))-dlog10m, Mmax=np.log10(np.max(ms))+dlog10m, dlog10m=dlog10m, hmf_model=self.model, mdef_model="SOCritical", mdef_params = {"overdensity": 200}) for z in zs]  # TODO 2
        HMF_m_z = np.array([np.interp(ms, haloMF.m, haloMF.dndlog10m) for haloMF in haloMFsraw]).T
        return HMF_m_z


Classes = {
    "hmf_package": hmf_package,
}

def get_Class(class_name):
    print("Loading HMF")
    try:
        return Classes[class_name]
    except KeyError:
        raise ValueError(f"Unknown class: {class_name}. Choose from {list(Classes.keys())}")