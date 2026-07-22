"""
DESI luminous red galaxy samples for cross-correlations

ui.adsabs.harvard.edu/abs/2023JCAP...11..097Z
arxiv.org/pdf/2309.06443
"""


from config import *

from scipy.special import erf

from Models.Papers.PlotsTables import BasePlots2, splittable, ParamTable
thispath = os.path.dirname(os.path.abspath(__file__))

    # subs = {
    #     'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
    #     'sample' : ['main', 'ext'],  # sample of LRGs
    #     'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    # }
    # info = {
    #     'area': {  # area of survey [deg^2]], 2.1p2/3.2p3
    #             'main':{'combined':16700, 'north':4200, 'south':12500},
    #             'extended':{'combined':230, 'north':100, 'south':130}},
    #     'logMhMean': {'z1': 13.40, 'z2': 13.40, 'z3': 13.24, 'z4': 13.24},  # mean halo mass taken from Yuan 2023 [Msun?], 6.p2
    #     # mean number density, mean redshift, min/max photometrix redshift bounds,  T1/T2/T3
    #     'nGal': {  # [deg^-2]
    #         'main': {'all':600, 'z1': 81.9, 'z2': 148.1, 'z3': 162.4, 'z4': 148.3},
    #         'ext': {'all':1669, 'z1': 185.5, 'z2': 311.0, 'z3': 422.6, 'z4': 438.4},},
    #     'zMean': {
    #         'main': {'z1': 0.470, 'z2': 0.628, 'z3': 0.791, 'z4': 0.924},
    #         'ext': {'z1': 0.467, 'z2': 0.633, 'z3': 0.794, 'z4': 0.929},},
    #     'zpMin': {
    #         'main': {
    #             'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, 'z3': 0.719, 'z4': 0.851},
    #             'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, 'z3': 0.713, 'z4': 0.860},},
    #         'ext': {
    #             'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, '3': 0.719, 'z4': 0.854},
    #             'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, '3': 0.713, 'z4': 0.860},},},
    #     'zpMax': {
    #         'main': {
    #             'north': {'all':1.024, 'z1': 0.545, 'z2': 0.719, 'z3': 0.851, 'z4': 1.024},
    #             'south': {'all':1.020, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.020},},
    #         'ext': {
    #             'north': {'all':1.010, 'z1': 0.545, 'z2': 0.719, 'z3': 0.854, 'z4': 1.010},
    #             'south': {'all':1.000, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.000},},},
    # }
    # for val in ['zpMin', 'zpMax']:  # Assumeding combined uses south limits
    #     for samp in info[val].keys(): 
    #         info[val][samp]['combined'] = info[val][samp]['south']
    # info['area'] = cycle(info['area'], lambda a: a*u.deg**2)
    # info['nGal'] = cycle(info['nGal'], lambda n: n /u.deg**2)
    
    
class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(8, 6),
             xlabel=r'Redshift', xlim=(0.15, 1.25), xscale='linear',
             ylabel=r'$N (\text{deg}^{-2})$', ylim=(0, 28), yscale='linear'),
        dict(name='Fig2b', filename='Fig2b', figsize=(8, 6),
             xlabel=r'Redshift', xlim=(0.15, 1.25), xscale='linear',
             ylabel=r'$N (\text{deg}^{-2})$', ylim=(0, 75), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig3(BasePlots2):
    subplots = [[
        dict(name='Fig3a', filename='Fig3a', figsize=(8, 6),
             xlabel=r'Redshift', xlim=(0.15, 1.1), xscale='linear',
             ylabel=r'$10^{3} n(z) (h^{3} \ \text{Mpc}^{-3})$', ylim=(0, 0.65), yscale='linear'),
        dict(name='Fig3b', filename='Fig3b', figsize=(8, 6),
             xlabel=r'Redshift', xlim=(0.15, 1.1), xscale='linear',
             ylabel=r'$10^{3} n(z) (h^{3} \ \text{Mpc}^{-3})$', ylim=(0, 1.65), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # area (2.1p2/3.2p3) + halo mass (Yuan 2023, 6.p2) + T1/T2/T3, per sample/hemisphere/zbin
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        super().__init__(filename)


class Studies(BaseStudy):  # arxiv.org/abs/2309.06443
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }
    info = {}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        row = StudiesInfoTable().getparams(sample=self.sample, hemisphere=self.hemisphere, zbin=self.zbin).to_dict()
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v): v = None
            if k == 'area' and v is not None: v = v*u.deg**2
            if k == 'nGal' and v is not None: v = v/u.deg**2
            setattr(self, k, v)


class TargetData(BaseTargetData, Studies.Zhou2023):  # DESI LS DR9 LRGs for Cross-correlation (Zhou+ 2023, arxiv.org/abs/2309.06443)
    path = f"{datapath}/Zhou2023B"  # path to data from zenodo.org/records/8319955
    # rawpath = f'/global/cfs/projectdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/catalogs' # path to raw data from https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/
    # https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/

    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }

    def __init__(self, inputsdict={}, **inputvars):       
        self.setup(inputsdict | inputvars)

    def make_zdist(self, zMin=None, zMax=None, dz=None, zNum=None):
        self.require(['zbin', 'sample', 'hemisphere'])

        sampstr = 'extended_' if self.sample=='ext' else ''
        rawcat = Table.read(f"{self.path}/dr9_{sampstr}lrg_pzbins.fits")
        
        # spec = importlib.util.spec_from_file_location("cuts", f"{self.path}/quality_cuts.py")
        # cuts = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(cuts)
        # self.zs_df = cuts.apply_cuts(rawcat).to_pandas()
        
        self.finalcat = rawcat.to_pandas()
        zscat = self.finalcat[self.finalcat.Z_PHOT_MEDIAN>=0]
        zscat = zscat
        
        self.catdist('z', zscat, qMin=zMin, qMax=zMax, dq=dz, qNum=zNum, densspace=self.area)


class Measurements(BaseMeasurement, Studies.Zhou2023):  # DESI LS DR9 LRGs for Cross-correlation (Zhou+ 2023, arxiv.org/abs/2309.06443)
    path = f"{datapath}/Zhou2023B"  # path to data from zenodo.org/records/8319955
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }

    def __init__(self, inputsdict, **inputvars):       
        self.setup(inputsdict | inputvars)
        self.require(['zbin', 'sample', 'hemisphere'])

        self.get_meas()
        
    def get_meas(self):
        samp = {'main':'fid', 'ext':'ext'}[self.sample]

        Cls = json.load(open(f"{self.path}/combined_cls.json", "r"))
        self.ell = np.array(Cls['ell'])
        self.Cgg_data = np.array(Cls[f'cls_{samp}'][f's0{self.zbin[-1]}'])
        
        self.Wggs = json.load(open(f"{self.path}/combined_wth.json", "r"))
        self.thetas = np.array(self.Wggs['theta(deg)'])  # [deg]
        self.Wgg_data = np.array(self.Wggs[f'wth_{samp}'][f's0{self.zbin[-1]}'])


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2023JCAP...11..097Z
    def Fig2(self, width=16, height=6):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'Redshift', ylabel=r'$N (\text{deg}^{-2})$',
            xlim=(0.15, 1.25), ylim=[(0, 28), (0, 75)], xscale='linear', yscale='linear')
        
    def Fig3(self, width=16, height=6):
        return self.plot(filename=['Fig3a','Fig3b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'Redshift', ylabel=r'$10^{3} n(z) (h^{3} \ \text{Mpc}^{-3})$',
            xlim=(0.15, 1.1), ylim=[(0, 0.65), (0, 1.65)], xscale='linear', yscale='linear')
