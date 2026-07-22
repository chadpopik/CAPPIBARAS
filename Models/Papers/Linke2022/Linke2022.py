"""
KiDS+VIKING+GAMA: Halo occupation distributions and correlations of satellite numbers with a new halo model of the galaxy-matter bispectrum for galaxy-galaxy-galaxy lensing

ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
arxiv.org/pdf/2204.02418
"""


from config import *

from scipy.special import erf

from Models.Papers.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


class HOD():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


    def Ncen(self, pdict={}, **kwargs): # Eq 36
        p = self.p0 | pdict | kwargs
        return (p['alpha_a']/2) * (1+erf((self.logM-p['logMtha'])/p['sigmaa']))


    # subs = {'sample':['MS', 'KVG'],
    #         'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    # info = {
    #     # best fit params, unclear what for:
    #     'A': {'MS': {'r': 5.31, 'b': 5.31}, 'KVG': {'r': 1.62, 'b': 1.62}},
    #     'epsilon': {'MS': {'r': 0.69, 'b': 0.69}, 'KVG': {'r': 0.99, 'b': 0.99}},

    #     # Cosmo params 1.p8
    #     'Om0':{'MS': 0.25, 'KVG': 0.315}, 'Ob0':{'MS': 0.045, 'KVG': 0.049}, 'H0':{'MS': 73, 'KVG': 67.4}, 'sigma8':{'MS': 0.9, 'KVG': 0.811},
    #     'mdef': '200m', 'zMax': 0.5,
    #     # Msun/h^2, these cuts are just for sims
    #     'MhMin':{'MS': 10e11}, 'MhMax': {'MS':10e15},
    # }

class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(8, 6),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'$\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig8(BasePlots2):
    subplots = [[
        dict(name='Fig8a', filename='Fig8a', figsize=(5, 4),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'HOD $\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
        dict(name='Fig8b', filename='Fig8b', figsize=(5, 4),
             xlabel=r'Halo mass $m \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'HOD $\langle N^a|m\rangle$', ylim=(1e-4, 1e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # best fit params + Cosmo params, per sample (color-independent)
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)


class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
    subs = {'sample':['MS', 'KVG'],
            'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    info = {
        'mdef': '200m', 'zMax': 0.5,
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        row = StudiesInfoTable().getparams(sample=self.sample).to_dict()
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v): v = None
            if k=='H0' and v is not None: v = v *u.km/u.s/u.Mpc
            if k in ('MhMin', 'MhMax') and v is not None: v = v *u.Msun
            setattr(self, k, v)


class HODParamsTable(ParamTable):  # Best-fit parameters, Table 3
    def __init__(self, filename=f"{thispath}/hod_params.csv"):
        super().__init__(filename)


class HODs(BaseHOD, Studies.Linke2022):  # arxiv.org/abs/2204.02418
    models = {'sample':['MS', 'KVG', 'fid'],
            'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.check_inputs(inpdict=inputsdict | inputvars, optdict=self.models)
        self.p0 = HODParamsTable().getparams(sample=self.sample, color=self.color).to_dict()

    def Ncen(self, logM):  # Eq 36
        func = lambda p: Zheng2005().Nc(logM, logMmin=np.log10(p['M_th_a']*1e11), sigmalogM=p['sigma_a']) * p['alpha_a']
        return lambda p={}: func(self.p0 | p)

    def Nsat(self, logM):  # Eq 37
        func = lambda p: Zheng2005().Ns(10**logM/self.h/self.h, M0=0, M1 = p['M_a']*1e13, alpha=p['beta_a']) * self.Ncen(logM)(p)/p['alpha_a']
        return lambda p={}: func(self.p0 | p)

    def ns_r(self, rs, zs, logMs):
        rs, zs, logMs = self.setdim(rs, zs, logMs)
        rdels, cdels = self.r200m(zs, logMs), self.c200m(zs, logMs)
        xs = rs*u.Mpc/rdels
        NFW_func = 1/(xs*(1/cdels+xs)**2)
        A_NFW = (1+cdels)/(np.log(1+cdels)*(1+cdels)-cdels)
        rhom = self.rhoc(zs)*self.Omega_m
        val = 200*rhom/3 *NFW_func * A_NFW/(10**logMs*u.Msun)
        return lambda **kwargs: val

    def ns(self, ks, zs, logMs):
        self.require(['rhoc', 'c200m', 'r200m'])
        fft = HaloModels.mcfit_package(ks=ks)
        NFW = self.ns_r(fft.rs, zs, logMs)()
        val = fft.FFT3D(NFW)
        return lambda **kwargs: val


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
    def Fig2(self, width=8, height=6):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'Halo mass $m \ [M_\odot]$', ylabel=r'$\langle N^a|m\rangle$',
            xlim=(1e11, 1e15), ylim=(1e-4, 1e2), xscale='log', yscale='log')

    def Fig8(self, width=10, height=4):
        return self.plot(filename=['Fig8a','Fig8b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'Halo mass $m \ [M_\odot]$', ylabel=r'HOD $\langle N^a|m\rangle$',
            xlim=(1e11, 1e15), ylim=(1e-4, 1e2), xscale='log', yscale='log')
