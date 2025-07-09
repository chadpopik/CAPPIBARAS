"""
Collections of galaxy distributions in stellar mass and redshift for various survey samples.
Classes should have galaxy number density 2D arrays over stellar and redshift and the corresponding stellar mass/redshift arrays, in consistent units.
Some samples may need to combine a SMF from one study and a redshift distribution from another.
If available, using halo masses in m200c instead of stellar masses is fine, then just don't use a SHMR later.
"""

from Basics import *

class BaseSMF:
    def checkspefs(self, spefs, required):
        for mname in required:
            if spefs[mname] not in getattr(self, f"{mname}s"):  # Check if the model is in the list of models
                raise NameError(f"{mname} {spefs[mname]} doesn't exist, choose from available {mname}s: {getattr(self, f'{mname}s')}")
            else:
                setattr(self, mname, spefs[mname])
                
    def SMF_to_N(self, cosmopars):
        cosmo = astropy.cosmology.LambdaCDM(H0=cosmopars["hh"]*100, Tcmb0=cosmopars['T_CMB'], Om0=cosmopars["Omega_m"], Ode0=cosmopars["Omega_L"], Ob0=cosmopars["Omega_b"])
        
        vols = np.array([(cosmo.comoving_volume(z+0.1).value-cosmo.comoving_volume(z).value)/(1+z+0.05)**3 for z in self.zs])
        
        return vols*np.log10(self.mstars[1]/self.mstars[0])


class Gao2023(BaseSMF):  # DESI 1% LRGs and ELGs (arxiv.org/abs/2306.06317)
    zbins = ['all', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
    samples = ['LRG', 'ELG']
    hemispheres = ['combined', 'north', 'south']
    zweights = ['True', 'False']
    
    path = "/pscratch/sd/c/cpopik/save_data_point_DESI-2023-0213"
    redshift_dist_file = "/global/cfs/projectdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/redshift_dist/main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"
    surveyarea = 16700
    
    def __init__(self, spefs):
        self.checkspefs(spefs, required=['sample', 'zbin', 'hemisphere', 'zweight'])

        if self.sample=='LRG': self.zs = np.arange(0.4, 1.1, 0.1)
        elif self.sample=='ELG': self.zs = np.arange(0.6, 1.6, 0.1)

        self.mstars = 10**pd.read_csv(f"{self.path}/Fig1_{self.sample}_z0.8.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.SMFraw = np.array([pd.read_csv(f"{self.path}/Fig1_{self.sample}_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in self.zs]).T  # [(Mpc/h)^-3 dex^-1]
        
        # Read in the redshift distribution for each specific bin
        # Number density in N / deg^2, dataframe with 1D arrays of length [ndim_zs_zdist]
        self.zdistdf = pd.read_csv(self.redshift_dist_file, sep=" ", skiprows=1, names=pd.read_csv(self.redshift_dist_file, sep=" ").columns[1:])
        self.zdist = self.zdistdf[f"{self.zbin}_{self.hemisphere}"]
        
        # Sort/group/sum up the finer redshift bins to match the bins of the SMF distribution
        self.zdistdf['zbin'] = pd.cut(self.zdistdf['zmin'], bins=np.arange(self.zs[0], self.zs[-1]+0.2, 0.1))
        self.zdistscale = self.zdistdf.groupby('zbin')[f"{self.zbin}_{self.hemisphere}"].sum().values*self.surveyarea
        
    def get_gdist(self, cosmopars):
        self.gdist = self.SMFraw*cosmopars["hh"]**3*self.SMF_to_N(cosmopars)
        if self.zweight is True:
            self.gdist = self.gdist*self.zdistscale/np.sum(self.gdist, axis=0)
        self.zave = np.sum(self.gdist*self.zs)/np.sum(self.gdist)
        return self.gdist
    
    def get_SMF(self, cosmopars):
        self.SMF = self.SMFraw*cosmopars['hh']**3
        if self.zweight is True:
            return self.SMF*self.zdistscale/self.SMF_to_N(cosmopars)/np.trapz(self.SMF, np.log10(self.mstars), axis=0)
        return self.SMF
    


class DR10CMASS(BaseSMF):
    groups = ['portsmouth', 'wisconsin', 'granada']
    IMFs = ['krou', 'salp']
    templates = ['starforming', 'passive']
    pops = ['bc03', 'm11']
    times = ['earlyform', 'wideform']
    dusts = ['dust', 'nodust']
    
    path = "/global/cfs/projectdirs/sdss/data/sdss/dr10/boss/spectro/redux/galaxy/v1_0"
    skyfrac = (6373.2/41253) 

    def __init__(self, spefs):
        self.checkspefs(spefs, required=['group'])
        if self.group=='portsmouth': 
            self.checkspefs(spefs, required=['template', 'IMF'])
            fname, mcolname = f"stellarmass_{self.template}_{self.IMF}", "LOGMASS"
        elif self.group=='wisconsin': 
            self.checkspefs(spefs, required=['pop'])
            fname, mcolname = f"pca_{self.pop}", "MSTELLAR_MEDIAN"
        elif self.group=='granada': 
            self.checkspefs(spefs, required=['IMF', 'time', 'dust'])
            fname, mcolname = f"fsps_{self.IMF}_{self.time}_{self.dust}", "MSTELLAR_MEDIAN"


        self.dfdata = Table.read(f"{self.path}/{self.group}_{fname}-v5_5_12.fits.gz")['Z', mcolname].to_pandas().rename(columns={mcolname: "LOGM"})

        mbins = np.arange((np.floor(self.dfdata.LOGM.min()*10)/10).round(1), (np.ceil(self.dfdata.LOGM.max()*10)/10+0.1).round(1), 0.1)
        zbins = np.arange((np.floor(self.dfdata.Z.min()*10)/10).round(1), (np.ceil(self.dfdata.Z.max()*10)/10+0.1).round(1), 0.1)
        
        self.gdist, self.mstars, self.zs = np.histogram2d(self.dfdata.LOGM, self.dfdata.Z, bins=[mbins, zbins])
        self.mstars, self.zs = 10**self.mstars[:-1], self.zs[:-1]

    def get_SMF(self, cosmopars):
        self.SMF = self.gdist/self.SMF_to_N(cosmopars)
        return self.SMF

    def get_gdist(self, cosmopars={}):
        self.zave = np.sum(self.gdist*self.zs)/np.sum(self.gdist)
        return self.gdist
        



    
    
# class Zhou2022:  # DESI LRGs (arxiv.org/abs/2208.08515)
#     path = "/global/cfs/projectdirs/desi/public/ets/vac/stellar_mass/v1/"


# class Amodeo2021():
#     def __init__(self):
#         self.allms = np.genfromtxt("/global/homes/c/cpopik/Git/Capybara/Data/mass_distrib.txt")
#         mbins = np.arange((np.floor(self.allms.min()*10)/10).round(1), (np.ceil(self.allms.max()*10)/10+0.1).round(1), 0.1)
#         self.SMF, self.mstars = np.histogram(self.allms, bins=mbins)
        
#         self.zs = 


# Classes = {
#     "Gao2023": Gao2023,
#     "DR10CMASS": DR10CMASS,
# }

# def get_Class(survey_name):
#     print("Loading SMF")
#     try:
#         return Classes[survey_name]
#     except KeyError:
#         raise ValueError(f"Unknown Sample: {survey_name}. Choose from {list(Classes.keys())}")