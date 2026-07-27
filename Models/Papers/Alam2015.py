"""
The Eleventh and Twelfth Data Releases of the Sloan Digital Sky Survey: Final Data from SDSS-III


ui.adsabs.harvard.edu/abs/2015ApJS..219...12A
arxiv.org/pdf/1501.00963
"""

from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Alam2015")

class Data():
    # The total footprint is about 10,400 deg2 (Figure 6); the value of 9376 deg2in Table 1 excludes masked regions due to bright stars and data that do not meet our survey requirements.
    area = 9376 * u.deg**2
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)




"""Everything below this line is old"""



from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # The Eleventh and Twelfth Data Releases of the Sloan Digital Sky Survey: Final Data from SDSS-III, ui.adsabs.harvard.edu/abs/2015ApJS..219...12A
    subs = {'DR': ['DR10', 'DR12']
    }
    info = {
        'area': {'DR12': 9376, 'DR10': 6373.2},
        }

    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)
    
# class SDSSBOSS(BaseTargetData, Studies.Ahn2013Alam2015):  # (Ahn+ 2013, arxiv.org/abs/1307.7735, Alam+ 2015, https://arxiv.org/abs/1501.00963)
#     path = '/global/cfs/projectdirs/sdss/data/sdss'   # Path of the data in NERSC
#     if not os.path.isdir(path): # Path through a URL if not in NERSC
#         path = 'https://data.sdss.org/sas/'
#     subs = {
#         'DR': ['DR10', 'DR12'],
#         'galaxy': ['CMASS', 'LOWZ'],  # Galaxy sample
#         'group': ['portsmouth', 'wisconsin', 'granada'],  # Group models
#         'IMF': ['krou', 'salp'],  # Kroupe or Salpeter Initial Mass Function
#         'template': ['starforming', 'passive'],
#         'pop': ['bc03', 'm11'], # Bruzual-Charlot or Maraston population
#         'time': ['earlyform', 'wideform'],  # early or extended SF
#         'dust': ['dust', 'nodust'],
#     }

#     def __init__(self, inputsdict, **inputvars):
#         self.setup(inputsdict | inputvars)
#         self.require(['DR'])
        
#         self.get_catalog()
        
#     def get_catalog(self):  # import catalog
#         self.require(['group'])  # base info
#         vers = {'DR10':['v1_0', '5_12'], 'DR12':['v1_1', '7_0']}[self.DR]
#         self.path = f"{self.path}/dr{self.DR[2:]}/boss/spectro/redux/galaxy/{vers[0]}"  # locate folder
            
#         if self.group=='portsmouth': 
#             self.require(['template', 'IMF'])
#             fname, mcolname = f"{self.group}_stellarmass_{self.template}_{self.IMF}-v5_{vers[1]}.fits.gz", 'LOGMASS'
#         elif self.group=='wisconsin': 
#             self.require(['pop'])
#             fname, mcolname = f"{self.group}_pca_{self.pop}-v5_{vers[1]}.fits.gz", 'MSTELLAR_MEDIAN'
#         elif self.group=='granada': 
#             self.require(['template', 'time', 'dust'])
#             fname, mcolname = f"{self.group}_fsps_{self.IMF}_{self.time}_{self.dust}-v5_{vers[1]}.fits.gz", 'MSTELLAR_MEDIAN'

#         # Fetch the data and rename the mass column, sdss4.org/dr17/spectro/galaxy_portsmouth/
#         self.dfdata = Table.read(f"{self.path}/{fname}")['Z', mcolname, 'BOSS_TARGET1'].to_pandas().rename(columns={mcolname: "LOGM"})

#         # Select the correct galaxy sample using the bitmasks in sdss3.org/dr10/algorithms/bitmask_boss_target1.php, sdss3.org/dr10/algorithms/bitmasks.php, sdss4.org/dr17/algorithms/bitmasks/, skyserver.sdss.org/dr19/MoreTools/browser/
#         decode_bitmask = lambda val: [i for i in range(val.bit_length()) if (val >> i) & 1]
#         self.dfdata['bits'] = self.dfdata['BOSS_TARGET1'].apply(decode_bitmask)
#         self.bitmasks = {'CMASS':7, 'LOWZ':0}

#     def make_dNdz(self, dz=0.1, zMin=None, zMax=None, **kwargs):
#         self.require(['galaxy'])
#         dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
#         self.N_z, self.z, zbins = self.make_N_q(dfdata.Z, dz, zMin, zMax)
#         self.dz = dz
#         self.dNdz = self.N_z / self.dz/u.dex
#         self.dndz = self.dNdz/ self.area/u.deg**2
    
#     def get_dNdlogMs(self, dlogMs=0.05, logMsmin=None, logMsmax=None):
#         self.require(['galaxy'])
#         dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
#         dNdlogMs, logMs, logMsbins = self.make_N_q(dfdata.LOGM, dlogMs, logMsmin, logMsmax)
#         return dNdlogMs, logMs, dlogMs
        
#     def get_dNdlogMs(self, dlogMs=0.05, dz=0.05, zmin=None, zmax=None, logMsmin=None, logMsmax=None):
#         self.require(['galaxy'])
#         dfdata = self.dfdata[self.dfdata["bits"].apply(lambda bits: (self.bitmasks[self.galaxy] in bits))]
#         dNdlogMsdz, dNdlogMs, dNdz, logMs, z = self.make_N_q1_q2(dfdata.LOGM, dlogMs, self.dfdata.Z, dz)
#         dNdlogMs2D = dNdlogMsdz*dz*u.dex
#         return dNdlogMs2D, z, logMs, dz, dlogMs