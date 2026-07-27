"""
Backlighting extended gas halos around luminous red galaxies: Kinematic Sunyaev-Zel'dovich effect from DESI Y1 and ACT data

ui.adsabs.harvard.edu/abs/2025PhRvD.112j3512R
arxiv.org/pdf/2503.19870
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "RiedGuachalla2025")

from scipy.special import erf



class Data():
    # III.A The DESI LRG Y1 galaxies overlapping with ACT are distributed in ∼4,300 deg2
    area = 4300 *u.deg**2
    
    
class Measurements_new():
    path = f"{DATA_PATH}/RiedGuachalla2025"  # zenodo.org/records/15081008

    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

    def get_meas(self):
        # TODO: Add more of the plots as options

        kSZ_fid = dict(np.load(f"{self.path}/fig8_fiducial.npz")) # The measured stacked kSZ in μK arcmin2 for varying CAP filters with radius R
        kSZ_zbins = dict(np.load(f"{self.path}/fig11_ksz_z.npz"))  # Mean stacked kSZ profiles for the different redshift bins
        kSZ_mbins = dict(np.load(f"{self.path}/fig12_ksz_mass.npz"))  # kSZ stacked profiles for the different stellar mass bins, denoted by mass

        if self.bin=='all':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_fid['R'] *u.arcmin, kSZ_fid['DESIxACT'] *u.uK*u.arcmin**2, kSZ_fid['errors_DESIxACT'] *u.uK*u.arcmin**2

        elif self.bin[0]=='z':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_zbins['R'] *u.arcmin, kSZ_zbins[f'{self.bin}'] *u.uK*u.arcmin**2, kSZ_zbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

        elif self.bin[0]=='m':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_mbins['R'] *u.arcmin, kSZ_mbins[f'{self.bin}'] *u.uK*u.arcmin**2, kSZ_mbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

        self.cormat = dict(np.load(f"{self.path}/fig18_cor.npz"))['cor']
        self.TkSZ_cov = np.diag(self.TkSZ_err) @ self.cormat @ np.diag(self.TkSZ_err)
        


# TABLE II: Statistics of the DESI LRG Y1 sample kSZ and across various bins when splitting it by redshift (denoted as z followed by the bin number), stellar mass (denoted as mass followed by the bin number), and absolute magnitude for the three optical bands (denoted as Mag- followed by the color and bin number). For a given bin and total sample, we report the mean redshift ⟨z⟩, the median redshift Med(z), the mean stellar mass ⟨M⋆⟩ in units of solar masses (M⊙) and the number of galaxies N . We also include χ2 null from Eq. 9 with nine degrees of freedom, which quantifies the rejection of the null hypothesis, and the corresponding S/N (Eqs. 10). Finally, we included the amplitudes and uncertainties shown in Fig. 10. For the complete sample, we find S/N = 9.8.
# III.B we split the DESI LRG Y1 overlapping with ACT into four spectroscopic redshift bins:(0.4, 0.6), (0.6, 0.8), (0.8, 0.95) and (0.95, 1.1)....Additionally, following [Hadzhiyska 2025], we split the stellar masses into four bins of log10(M∗/M⊙): (10.5, 11.2), (11.2, 11.4), (11.4, 11.6), and (11.6, 12.5)
class Table2(ParamTable):
    def __init__(self, filename=f"{thispath}/Table2.csv"):
        self.df = pd.read_csv(filename)


# FIG. 2: Redshift distribution of the DESI LRG Y1 galaxies overlapping the ACT map (∼ 39% of the total LRG sample). The galaxies were divided into four bins given their spectroscopic redshift, shown by the vertical lines. Additional summary information, including mean redshift and number of objects per bin, is included in Table II.
class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(6, 5),
             xlabel=r'$z$', xlim=(0.365, 1.135), xscale='linear',
             ylabel=r'Number', ylim=(0, 35e3), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 3: The binned stellar mass distribution of the DESI LRG Y1 galaxies overlapping the ACT DR6 map, as estimated by [85], is provided. Additional summary information for the bins is available in Table II.
class Fig3(BasePlots2):
    subplots = [[
        dict(name='Fig3', filename='Fig3', figsize=(6, 5),
             xlabel=r'Stellar Mass $M_* \ [M_\odot]$', xlim=(2.6e10, 3.5e12), xscale='log',
             ylabel=r'Number', ylim=(5.5e-1, 1.5e5), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 8: The measured stacked kSZ in μK arcmin2 for varying CAP filters with radius R from Eq. 8 in brown. We include the corresponding comoving distances atz = 0.8 on the top horizontal axis. Several simulation CAP profiles are included, each rescaled in amplitude to match the measured signal at the largest aperture where all the baryons should be encompassed (upper panel), or with the amplitude left free to facilitate a comparison of their shapes in the presence of mass mismatch between data and simulation (lower panel). For the IllustrisTNG case (solid blue, labeled TNG in the figure), the profile shape more closely follows that of dark matter than the observed data. For Illustris (light blue dashed) atz = 0.8, the profiles tend to align more closely with the observations. In contrast, when comparing the Illustris profile at z = 0.5 (light green dashed), taken from [42], we find a better match with the shape of the kSZ profile. The bands on the lower panel propagate the uncertainty on the profile amplitude from Eq. 15. The vertical gray line shows the virial radius added in quadrature with the beam standard deviation (FWHM/p8 ln(2)) and the secondary axis on the right translates to the integrated optical depth of Thomson scattering.
class Fig8(BasePlots2):
    subplots = [[
        dict(name='Fig8a', filename='Fig8a', figsize=(5, 3.5),
             xlabel=r'$R \ [\text{arcmin}]$', xlim=(0.525, 6.525), xscale='linear',
             ylabel=r'$T_\text{kSZ} \ [\mu \text{K arcmin}^2]$', ylim=(0.1, 20), yscale='log'),
        dict(name='Fig8b', filename='Fig8b', figsize=(5, 3.5),
             xlabel=r'$R \ [\text{arcmin}]$', xlim=(0.525, 6.525), xscale='linear',
             ylabel=r'$T_\text{kSZ} \ [\mu \text{K arcmin}^2]$', ylim=(0.1, 20), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 11: Mean stacked kSZ profiles for the different redshift bins. There is no clear trend of the profiles, confirming the results obtained by [42]. Additionally, the number of galaxies in the redshift bins is not evenly distributed: z4 reports the lowest S/N = 2.3 with 96,346 galaxies, which e.g. corresponds to only ∼ 1/3 of the galaxies of bin z2.
class Fig11(BasePlots2):
    subplots = [[
        dict(name='Fig11', filename='Fig11', figsize=(5, 3.5),
             xlabel=r'$R \ [\text{arcmin}]$', xlim=(0.75, 6.4), xscale='linear',
             ylabel=r'$T_\text{kSZ} \ [\mu \text{K arcmin}^2]$', ylim=(0.045, 35), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 12: Similar to Fig. 11, the kSZ stacked profiles for the different stellar mass bins, denoted by mass. For mass4, there is a clear increment on the amplitude, similar to what [42] found. At larger scales, the errors are wider and more correlated, thus the data at largeR provides little new information beyond that from the smaller scales.
class Fig12(BasePlots2):
    subplots = [[
        dict(name='Fig12', filename='Fig12', figsize=(5, 3.5),
             xlabel=r'$R \ [\text{arcmin}]$', xlim=(0.75, 6.4), xscale='linear',
             ylabel=r'$T_\text{kSZ} \ [\mu \text{K arcmin}^2]$', ylim=(0.065, 28), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 20: The redshift distribution of the DESI LRG Y1 is shown for both the North Galactic Cap (NGC) and South Galactic Cap (SGC), along with the BOSS data. The NGC and SGC samples contain 1,476,132 and 662,468 galaxies, respectively. For comparison, the CMASS and LOWZ samples from BOSS consist of 777,202 and 218,905 galaxies, respectively, and correspond to lower redshifts. We emphasize that these are not the galaxies overlapping with the ACT DR6 map.
class Fig20(BasePlots2):
    subplots = [[
        dict(name='Fig20', filename='Fig20', figsize=(6, 4.5),
             xlabel=r'$z$', xlim=(0.1, 1.15), xscale='linear',
             ylabel=r'Number', ylim=(0, 60e3), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)





"""Old implementation being phased out"""

from Models.Studies import BaseStudy, cycle
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112j3512R
    subs = {}  # subset of galaxy selection
    info = {
        }
    
    
class Measurements(Study):  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs (arxiv.org/abs/2503.19870)
    path = f"{DATA_PATH}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    subs = {
        'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],  # redshift/mass bin
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['bin'])

        self.get_meas()  # get the measurements

    def get_meas(self):
        # TODO: Add more of the plots as options
        
        kSZ_fid = dict(np.load(f"{self.path}/fig8_fiducial.npz")) # The measured stacked kSZ in μK arcmin2 for varying CAP filters with radius R
        kSZ_zbins = dict(np.load(f"{self.path}/fig11_ksz_z.npz"))  # Mean stacked kSZ profiles for the different redshift bins
        kSZ_mbins = dict(np.load(f"{self.path}/fig12_ksz_mass.npz"))  # kSZ stacked profiles for the different stellar mass bins, denoted by mass

        if self.bin=='all':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_fid['R'] *u.arcmin, kSZ_fid['DESIxACT'] *u.uK*u.arcmin**2, kSZ_fid['errors_DESIxACT'] *u.uK*u.arcmin**2

        elif self.bin[0]=='z':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_zbins['R'] *u.arcmin, kSZ_zbins[f'{self.bin}'] *u.uK*u.arcmin**2, kSZ_zbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

        elif self.bin[0]=='m':
            self.R, self.TkSZ_data, self.TkSZ_err = kSZ_mbins['R'] *u.arcmin, kSZ_mbins[f'{self.bin}'] *u.uK*u.arcmin**2, kSZ_mbins[f'{self.bin}_error'] *u.uK*u.arcmin**2

        self.cormat = dict(np.load(f"{self.path}/fig18_cor.npz"))['cor']
        self.TkSZ_cov = np.diag(self.TkSZ_err) @ self.cormat @ np.diag(self.TkSZ_err)
        
        
        
from Models.TargetData import BaseTargetData
from Models.Papers import Gao2023
class TargetData(BaseTargetData, Study):  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs
    path = f"{DATA_PATH}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    subs = {
        'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],  # redshift/mass subsample
    }
    
    info = {
        'area': 4300 *u.deg**2,  # overlapping region of ACT and DESI [deg^2], F1 and III.B.p4
        'logMhMean': 13.4 *u.dimensionless_unscaled,  # Msun/littleh, estimated mean halo mass of LRG [Msun/h], III.B.p5
        'zMin': {# spectroscopic redshift bins, III.B.p6
            'all':0.4, 'z_1':0.4, 'z_2':0.6, 'z_3':0.8, 'z_4':0.95},
        'zMax': {# spectroscopic redshift bins, III.B.p6
            'all':1.1, 'z_1':0.6, 'z_2':0.8, 'z_3':0.95, 'z_4':1.1},
        'logMsMin': {# stellar mass bins [Msun], III.B.p7
            'all':10.5, 'mass_1':10.5, 'mass_2':11.2, 'mass_3':11.4, 'mass_4':11.6},
        'logMsMax': {# stellar mass bins [Msun], III.B.p7
            'all':12.40, 'mass_1':11.2, 'mass_2':11.4, 'mass_3':11.6, 'mass_4':12.5},
        'zMean': {# mean redshift, T2
            'all':0.74, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.76, 'mass_2':0.75, 'mass_3':0.71, 'mass_4':0.69},
        'zMed': {# median redshift, T2
            'all':0.75, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.79, 'mass_2':0.76, 'mass_3':0.70, 'mass_4':0.67},
        'MsMean': {'units': 1e11*u.Msun, # mean stellar mass , T2
            'all':2.2, 'z_1':2.4, 'z_2':2.3, 'z_3':2.0, 'z_4':2.1, 'mass_1':1.2, 'mass_2':2.0, 'mass_3':3.0, 'mass_4':5.1},
        'NGal': {# number of galaxies, T2
            'all':825283, 'z_1':195877, 'z_2':235620, 'z_3':235620, 'z_4':96346, 'mass_1':244932, 'mass_2':320914, 'mass_3':194037, 'mass_4':53997},
    }
    
    # Assume z/m bins have same m/z limits as all
    for b in [1, 2, 3, 4]:
        info['zMin'][f'mass_{b}'] = info['zMin']['all']
        info['zMax'][f'mass_{b}'] = info['zMax']['all']
        info['logMsMin'][f'z_{b}'] = info['logMsMin']['all']
        info['logMsMax'][f'z_{b}'] = info['logMsMax']['all']
    
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        
    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None, **kwargs):
        self.require(['bin'])  # require bin for file specification
        self.allcat_zs = dict(np.load(f"{self.path}/fig2_hist_z.npz"))  # load redshifts of all galaxies in catalog
        if self.bin[0]=='z': self.cat_zs = self.allcat_zs[f'{self.bin}']  # if zbin is specified, select for that subsample
        else: self.cat_zs = np.concatenate([self.allcat_zs[f'z_{i}'] for i in range(1, 5)])  # otherwise, get all of them

        self.catdist('z', self.cat_zs, qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)
        
    def make_Msdist(self, halomodel, logMsMin=None, logMsMax=None, dlogMs=None, logMsNum=None, **kwargs):
        self.require(['bin'])  # require bin for file specification
        self.allcat_Ms = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # load stellar masses of all galaxies from file
        if self.bin[0]=='m': self.cat_Ms = self.allcat_Ms[f'{self.bin}']  # if mbin is specified, select for that subsample
        else: self.cat_Ms = np.concatenate([self.allcat_Ms[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
        self.cat_logMs = np.log10(self.cat_Ms)  # convert to log values
        
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.zMax)-halomodel.Vcom(self.zMin))/(1+self.zMean)**3
        
        self.catdist('logMs', self.cat_logMs, qMin=logMsMin, qMax=logMsMax, dq=dlogMs, qNum=logMsNum, densspace=vol)
        
    def make_Mhdist(self, halomodel, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None, **kwargs):
        self.require(['bin'])  # require bin for file specification
        self.allcat_Ms = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # load stellar masses of all galaxies from file
        if self.bin[0]=='m': self.cat_Ms = self.allcat_Ms[f'{self.bin}']  # if mbin is specified, select for that subsample
        else: self.cat_Ms = np.concatenate([self.allcat_Ms[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
        self.cat_logMs = np.log10(self.cat_Ms)  # convert to log values
        
        # Get function to convert stellar masses to halo masses
        self.cat_logMh = Gao2023.SHMR({'model':'Psat'}).HSMR(self.cat_logMs)()  # Use SHMR of Gao 2023
        # TODO: 1. is the Psat model the best to use?, 2. think this converts to virial mass
        
        vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(halomodel.Vcom(self.zMax)-halomodel.Vcom(self.zMin))/(1+self.zMean)**3

        self.catdist('logMh', self.cat_logMh, qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=vol)


    # TODO: add 2D z/M distribution?