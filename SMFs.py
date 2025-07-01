"""
Collections of galaxy distributions in stellar mass and redshift for various survey samples.
Classes should have galaxy number density 2D arrays over stellar and redshift and the corresponding stellar mass/redshift arrays, in consistent units.
Some samples may need to combine a SMF from one study and a redshift distribution from another.
If available, using halo masses in m200c instead of stellar masses is fine, then just don't use a SHMR later.


TODO: Check for units and factors of h/dex
TODO: Clean CMASS DR10 and making adaptable to ms and zs
TODO: Make the class more organized and sensible with functionality
"""

from Basics import *


class Gao2023:  # DESI 1% LRGs and ELGs (arxiv.org/abs/2306.06317)
    mass_dist_dir = "/pscratch/sd/c/cpopik/save_data_point_DESI-2023-0213"
    redshift_dist_file = "/global/cfs/projectdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/redshift_dist/main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"
    bins = ['all', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
    hemispheres = ['combined', 'north', 'south']
    
    def __init__(self, bin='all', hemisphere='combined'):
        self.zs_LRG = np.arange(0.4, 1.1, 0.1)
        self.zs_ELG = np.arange(0.6, 1.6, 0.1)

        self.mstars = 10**pd.read_csv(f"{self.mass_dist_dir}/Fig1_LRG_z0.4.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[0]).Mstar.values  # [M_sol]
        self.SMF_LRG = np.array([pd.read_csv(f"{self.mass_dist_dir}/Fig1_LRG_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in self.zs_LRG]).T  # [(Mpc/h)^-3 dex^-1]
        self.SMF_ELG = np.array([pd.read_csv(f"{self.mass_dist_dir}/Fig1_ELG_z{z:.1f}.txt", sep=' ', names=['Mstar',f"n", f"err"], usecols=[1]).n.values for z in self.zs_ELG]).T  # [(Mpc/h)^-3 dex^-1]

        # Read in the redshift distribution for each specific bin
        # Number density in N / deg^2, dataframe with 1D arrays of length [ndim_zs_zdist]
        self.zdistdf = pd.read_csv(self.redshift_dist_file, sep=" ", skiprows=1, names=pd.read_csv(self.redshift_dist_file, sep=" ").columns[1:])
        self.zdist = self.zdistdf[f"{bin}_{hemisphere}"]
        # Sort/group/sum up the finer redshift bins to match the bins of the SMF distribution
        self.zdistdf['zbin'] = pd.cut(self.zdistdf['zmin'], bins=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        # Scale the SMF to match the redshift distribution of the bin in question
        self.SMF = self.SMF_unweighted*self.zdistdf.groupby('zbin')[f"{bin}_{hemisphere}"].sum().values/np.trapz(self.SMF_unweighted, self.mstars, axis=0)



class DR10CMASS:
    path = "/global/cfs/projectdirs/sdss/data/sdss/dr10/boss/spectro/redux/galaxy/v1_0"
    groups = {'portsmouth': {'template': ['starforming', 'passive'], 'IMF': ['krou', 'salp']},
              'wisconsin': {'pop': ['bc03', 'm11'], 'IMF': 'krou'},
              'granada': {'IMF': ['krou', 'salp'], 'time': ['earlyform', 'wideform'], 'dust': ['dust', 'nodust']}}
    skyfrac = (6373.2/41253) 
    cosmo = Planck18

    def __init__(self, group, *models):
        if group=='portsmouth': fname, mcolname = self.portsmouth(*models)
        elif group=='wisconsin': fname, mcolname = self.wisconsin(*models)
        elif group=='granada': fname, mcolname = self.granada(*models)
        else: raise NameError(f"Group {group} doesn't exist, choose from available samples: {list(self.groups.keys())}")

        self.dfdata = Table.read(f"{self.path}/{group}_{fname}-v5_5_12.fits.gz")['Z', mcolname].to_pandas().rename(columns={mcolname: "LOGM"})

        mbins = np.arange((np.floor(self.dfdata.LOGM.min()*10)/10).round(1), (np.ceil(self.dfdata.LOGM.max()*10)/10+0.1).round(1), 0.1)
        zbins = np.arange((np.floor(self.dfdata.Z.min()*10)/10).round(1), (np.ceil(self.dfdata.Z.max()*10)/10+0.1).round(1), 0.1)
        self.SMF, self.mstars, self.zs = np.histogram2d(self.dfdata.LOGM, self.dfdata.Z, bins=[mbins, zbins])
                                         
        vols = self.skyfrac * np.array([self.cosmo.comoving_volume(zbins[i+1]).value-self.cosmo.comoving_volume(zbins[i]).value for i in range(zbins.size-1)])
        self.SMF = self.SMF/vols
        
        self.mstars, self.zs = 10**self.mstars[:-1], self.zs[:-1]

        
    def portsmouth(self, template, IMF):
        return f"stellarmass_{template}_{IMF}", "LOGMASS"

    def wisconsin(self, pop):
        return f"pca_{pop}", "MSTELLAR_MEDIAN"

    def granada(self, IMF, time, dust):
        return f"fsps_{IMF}_{time}_{dust}", "MSTELLAR_MEDIAN"


# class Amodeo2021():
#     def __init__(self):
#         self.allms = np.genfromtxt("/global/homes/c/cpopik/Git/Capybara/Data/mass_distrib.txt")
#         mbins = np.arange((np.floor(self.allms.min()*10)/10).round(1), (np.ceil(self.allms.max()*10)/10+0.1).round(1), 0.1)
#         self.SMF, self.mstars = np.histogram(self.allms, bins=mbins)
        
#         self.zs = 


Classes = {
    "Gao2023": Gao2023,
    "DR10CMASS": DR10CMASS,
}

def get_Class(survey_name):
    print("Loading SMF")
    try:
        return Classes[survey_name]
    except KeyError:
        raise ValueError(f"Unknown Sample: {survey_name}. Choose from {list(Classes.keys())}")