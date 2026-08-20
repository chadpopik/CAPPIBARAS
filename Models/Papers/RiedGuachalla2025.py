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

    
    
class Measurement():  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs (arxiv.org/abs/2503.19870)
    path = f"{DATA_PATH}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    subs = {
        'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],  # redshift/mass bin
    }
    
    observable = "TkSZ"

    def __init__(self, bin, inputsdict={}, **inputvars):
        self.bin = bin

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
        
        
        
class TargetData():  # Stacked kSZ measurement of ACT DR6 and DESI Y1 LRGs
    path = f"{DATA_PATH}/RiedGuachalla2025"  # Path to data downloaded from zenodo.org/records/15081008
    
    area = 4300 *u.deg**2  # overlapping region of ACT and DESI [deg^2], F1 and III.B.p4
    logMhMean = 13.4 *u.dimensionless_unscaled  # Msun/littleh, estimated mean halo mass of LRG [Msun/h], III.B.p5
    v_rms = 233*u.km/u.s  # rms line-of-sight velocity of LRGs, estimated from the halo model and the linear matter power spectrum, III.B.p5

    def __init__(self, bin, 
                 inputsdict={}, **inputvars):
        self.bin = bin

        # Table II keys its rows as 'Full sample'/'z1'.../'mass1'... rather than our 'all'/'z_1'.../'mass_1'...
        csvbin = 'Full sample' if bin == 'all' else bin.replace('_', '')
        p = Table2().getparams(Bin=csvbin)
        # z bins don't quote stellar-mass limits, and mass/all bins don't quote redshift limits;
        # fall back to the full sample's limits (which catdist further falls back from if still missing)
        pall = Table2().getparams(Bin='Full sample') if bin != 'all' else p

        self.zMean, self.zMed, self.NGal = p['zmean'], p['zmed'], int(p['N'])
        self.MsMean = p['Mstar/1e11Msun'] * 1e11*u.Msun

        # only set these when the paper actually quotes a value for this bin; leave the attribute
        # undefined otherwise so downstream getattr(self, ..., <fallback>) calls fall back cleanly
        for attr, val in [
            ('zMin', p['zspecMin'] if not pd.isna(p['zspecMin']) else pall['zspecMin']),
            ('zMax', p['zspecMax'] if not pd.isna(p['zspecMax']) else pall['zspecMax']),
            ('logMsMin', p['logMstarMin'] if not pd.isna(p['logMstarMin']) else pall['logMstarMin']),
            ('logMsMax', p['logMstarMax'] if not pd.isna(p['logMstarMax']) else pall['logMstarMax']),
        ]:
            if pd.notna(val): setattr(self, attr, val)

        self.load_z_catalog()  # load redshifts of all galaxies in catalog
        self.load_Ms_catalog()
        
    def load_z_catalog(self):
        self.allcat_zs = dict(np.load(f"{self.path}/fig2_hist_z.npz"))  # load redshifts of all galaxies in catalog
        if self.bin[0]=='z': self.cat_zs = self.allcat_zs[f'{self.bin}']  # if zbin is specified, select for that subsample
        else: self.cat_zs = np.concatenate([self.allcat_zs[f'z_{i}'] for i in range(1, 5)])  # otherwise, get all of them
        
    def load_Ms_catalog(self):
        self.allcat_Ms = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # load stellar masses of all galaxies from file
        if self.bin[0]=='m': self.cat_Ms = self.allcat_Ms[f'{self.bin}']  # if mbin is specified, select for that subsample
        else: self.cat_Ms = np.concatenate([self.allcat_Ms[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
        self.cat_logMs = np.log10(self.cat_Ms)  # convert to log values
        
    def make_zbins(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        zMin = zMin if zMin is not None else getattr(self, 'zMin', self.cat_zs.min())
        zMax = zMax if zMax is not None else getattr(self, 'zMax', self.cat_zs.max())
        zNum = zNum if zNum is not None else np.int32((zMax-zMin)/dz)
        
        zbins = np.linspace(zMin, zMax, zNum)
        z = (zbins[1:]+zbins[:-1])/2
        dz = z[1]-z[0]
        return zbins, z, dz
        
    def N_z(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, zNum=zNum, dz=dz)
        return z, np.histogram(self.cat_zs, bins=zbins)[0]
    
    def dNdz(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, zNum=zNum, dz=dz)
        return z, np.histogram(self.cat_zs, bins=zbins)[0]/dz
    
    def n_z(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, zNum=zNum, dz=dz)
        return z, np.histogram(self.cat_zs, bins=zbins)[0]/self.area
    
    def dndz(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, zNum=zNum, dz=dz)
        return z, np.histogram(self.cat_zs, bins=zbins)[0]/dz/self.area
        
    def vol(self, cosmology):
        return (self.area/(4*np.pi*u.sr).to(u.deg**2))*(cosmology.Vcom(self.zMax)-cosmology.Vcom(self.zMin))/(1+self.zMean)**3

    def make_logMsbins(self, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, **kwargs):
        logMsMin = logMsMin if logMsMin is not None else getattr(self, 'logMsMin', self.cat_logMs.min())
        logMsMax = logMsMax if logMsMax is not None else getattr(self, 'logMsMax', self.cat_logMs.max())
        logMsNum = logMsNum if logMsNum is not None else np.int32((logMsMax-logMsMin)/dlogMs)

        logMsbins = np.linspace(logMsMin, logMsMax, logMsNum)
        logMs = (logMsbins[1:]+logMsbins[:-1])/2
        dlogMs = logMs[1]-logMs[0]
        return logMsbins, logMs, dlogMs

    def N_logMs(self, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, **kwargs):
        logMsbins, logMs, dlogMs = self.make_logMsbins(logMsMin=logMsMin, logMsMax=logMsMax, logMsNum=logMsNum, dlogMs=dlogMs)
        return logMs, np.histogram(self.cat_logMs, bins=logMsbins)[0]

    def dNdlogMs(self, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, **kwargs):
        logMsbins, logMs, dlogMs = self.make_logMsbins(logMsMin=logMsMin, logMsMax=logMsMax, logMsNum=logMsNum, dlogMs=dlogMs)
        return logMs, np.histogram(self.cat_logMs, bins=logMsbins)[0]/dlogMs

    def n_logMs(self, cosmology, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, **kwargs):
        logMsbins, logMs, dlogMs = self.make_logMsbins(logMsMin=logMsMin, logMsMax=logMsMax, logMsNum=logMsNum, dlogMs=dlogMs)
        return logMs, np.histogram(self.cat_logMs, bins=logMsbins)[0]/self.vol(cosmology)

    def dndlogMs(self, cosmology, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, **kwargs):
        logMsbins, logMs, dlogMs = self.make_logMsbins(logMsMin=logMsMin, logMsMax=logMsMax, logMsNum=logMsNum, dlogMs=dlogMs)
        return logMs, np.histogram(self.cat_logMs, bins=logMsbins)[0]/dlogMs/self.vol(cosmology)

    def add_Mh_catalog(self, conversion):
        if hasattr(self, 'logMsMin'):
            self.logMhMin = conversion(self.logMsMin)[0]
        if hasattr(self, 'logMsMax'):
            self.logMhMax = conversion(self.logMsMax)[0]
        if hasattr(self, 'logMsMean'):
            self.logMhMean = conversion(self.logMsMean)[0]
        self.cat_logMh = conversion(self.cat_logMs)

    def make_logMhbins(self, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, **kwargs):
        logMhMin = logMhMin if logMhMin is not None else getattr(self, 'logMhMin', self.cat_logMh.min())
        logMhMax = logMhMax if logMhMax is not None else getattr(self, 'logMhMax', self.cat_logMh.max())
        logMhNum = logMhNum if logMhNum is not None else np.int32((logMhMax-logMhMin)/dlogMh)

        logMhbins = np.linspace(logMhMin, logMhMax, logMhNum)
        logMh = (logMhbins[1:]+logMhbins[:-1])/2
        dlogMh = logMh[1]-logMh[0]
        return logMhbins, logMh, dlogMh

    def N_logMh(self, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, **kwargs):
        logMhbins, logMh, dlogMh = self.make_logMhbins(logMhMin=logMhMin, logMhMax=logMhMax, logMhNum=logMhNum, dlogMh=dlogMh)
        return logMh, np.histogram(self.cat_logMh, bins=logMhbins)[0]

    def dNdlogMh(self, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, **kwargs):
        logMhbins, logMh, dlogMh = self.make_logMhbins(logMhMin=logMhMin, logMhMax=logMhMax, logMhNum=logMhNum, dlogMh=dlogMh)
        return logMh, np.histogram(self.cat_logMh, bins=logMhbins)[0]/dlogMh

    def n_logMh(self, cosmology, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, **kwargs):
        logMhbins, logMh, dlogMh = self.make_logMhbins(logMhMin=logMhMin, logMhMax=logMhMax, logMhNum=logMhNum, dlogMh=dlogMh)
        return logMh, np.histogram(self.cat_logMh, bins=logMhbins)[0]/self.vol(cosmology)

    def dndlogMh(self, cosmology, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, **kwargs):
        logMhbins, logMh, dlogMh = self.make_logMhbins(logMhMin=logMhMin, logMhMax=logMhMax, logMhNum=logMhNum, dlogMh=dlogMh)
        return logMh, np.histogram(self.cat_logMh, bins=logMhbins)[0]/dlogMh/self.vol(cosmology)


    # def make_Msdist(self, cosmology, logMsMin=None, logMsMax=None, dlogMs=None, logMsNum=None, **kwargs):
    #     self.allcat_Ms = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # load stellar masses of all galaxies from file
    #     if self.bin[0]=='m': self.cat_Ms = self.allcat_Ms[f'{self.bin}']  # if mbin is specified, select for that subsample
    #     else: self.cat_Ms = np.concatenate([self.allcat_Ms[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
    #     self.cat_logMs = np.log10(self.cat_Ms)  # convert to log values
        
    #     vol = (self.area/(4*np.pi*u.sr).to(u.deg**2))*(cosmology.Vcom(self.zMax)-cosmology.Vcom(self.zMin))/(1+self.zMean)**3
        
    #     self.catdist('logMs', self.cat_logMs, qMin=logMsMin, qMax=logMsMax, dq=dlogMs, qNum=logMsNum, densspace=vol)
        
    # def make_Mhdist(self, cosmology=None, halomodel=None, MassDef=None, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None, **kwargs):
    #     self.allcat_Ms = dict(np.load(f"{self.path}/fig3_mass_dist.npz"))  # load stellar masses of all galaxies from file
    #     if self.bin[0]=='m': self.cat_Ms = self.allcat_Ms[f'{self.bin}']  # if mbin is specified, select for that subsample
    #     else: self.cat_Ms = np.concatenate([self.allcat_Ms[f'mass_{i}'] for i in range(1, 5)])  # otherwise, get all of them
    #     self.cat_logMs = np.log10(self.cat_Ms)  # convert to log values


    
    
    
    #     def catdist(self, qName, qs, qMin=None, qMax=None, dq=None, qNum=None, densspace=None):
    #     # Try to set range with input values first, then with info dict values, then just from data
    #     qMin = qMin if qMin is not None else ((getattr(self, f'{qName}Min', None)*u.dimensionless_unscaled).value if getattr(self, f'{qName}Min', None) is not None else qs.min())
    #     qMax = qMax if qMax is not None else ((getattr(self, f'{qName}Max', None)*u.dimensionless_unscaled).value if getattr(self, f'{qName}Max', None) is not None else qs.max())
    #     # Pretty ranges, may use later
    #     # if qmin is None: qmin = np.floor(qs.min()/dq)*dq  # default min
    #     # if qmax is None: qmax = np.ceil(qs.max()/dq)*dq  # default max
    #     # Try to get array size from input size of spacing
    #     try: qNum = qNum if qNum is not None else np.int32((qMax-qMin)/dq)
    #     except: raise ValueError("Must specify array spacing or length")

    #     setattr(self, f'{qName}Bins', np.linspace(qMin, qMax, qNum))  # make linearly spaced bins
    #     setattr(self, qName, (getattr(self, f'{qName}Bins')[1:]+getattr(self, f'{qName}Bins')[:-1])/2)  # center of bins
    #     setattr(self, f'd{qName}', getattr(self, qName)[1]-getattr(self, qName)[0])  # width of the bins
    #     setattr(self, f'N_{qName}', np.histogram(qs, bins=getattr(self, f'{qName}Bins'))[0])  # number distribution
    #     setattr(self, f'dNd{qName}', getattr(self, f'N_{qName}')/getattr(self, f'd{qName}'))  # differential number distribution
    #     if densspace is not None:  # if given area/volume, use that for the density
    #         setattr(self, f'n_{qName}', getattr(self, f'N_{qName}')/densspace)  # number density distribution
    #         setattr(self, f'dnd{qName}', getattr(self, f'dNd{qName}')/densspace)  # differential number density distribution


    # # TODO: make 2D dist from samples with two values
    # # def make_N_q1_q2(self, q1s, dq1, q2s, dq2, q1min=None, q1max=None, q2min=None, q2max=None):  # make 2D dist from samples with two values
    # #     N_q1, q1cents, q1bins = self.make_N_q(q1s, dq1, q1min, q1max)  # make 1D dist of value 1
    # #     N_q2, q2cents, q2bins = self.make_N_q(q2s, dq2, q2min, q2max)  # make 1D dist of value 1
    # #     N_q1_q2, _, _ = np.histogram2d(q1s, q2s, bins=[q1bins, q2bins])  # make 2D number dist from hist
    # #     # dNdq1dq2 = N /dq1/u.dex /dq2/u.dex  # apply units [dex^-2]
    # #     return N_q1_q2, N_q1, N_q2, q1cents, q2cents

    # def distave(self, q, qdist, qint=None):
    #     qint = qint if qint is not None else q
    #     return np.trapezoid(q*qdist, qint)/np.trapezoid(qdist, qint)