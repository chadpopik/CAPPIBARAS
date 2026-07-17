"""
Cosmological constraints from the tomographic cross-correlation of DESI Luminous Red Galaxies and Planck CMB lensing

ui.adsabs.harvard.edu/abs/2022JCAP...02..007W
arxiv.org/pdf/2111.09898
"""

import os
from Models.Papers.PlotsTables import ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))



    # subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'],} # photometric redshift subsmaple
    # info = {
    #     'SN': {'z1':4.02, 'z2':2.24, 'z3':2.07, 'z4':2.26},  # shot noise level [1e6]
    #     'zeff': {'z1':0.47, 'z2':0.62, 'z3':0.78, 'z4':0.91},  # effective z at which power spec is calculate approx
    #     'lmax': {'z1':250, 'z2':300, 'z3':350, 'z4':400},  # max ell used in fits
    #     'lSN': {'z1':400, 'z2':530, 'z3':575, 'z4':425},  # ell where SN equals modeled auto-spec power
    # }
    
    
        # def Fig2(self, width=8, height=4):
        # return self.plot(filename='Fig2', width=width, height=height,
        #     xlabel=r'$z$', ylabel=r'$d \text{ln} N/dz$ (actually $\frac{1}{N}\frac{dN}{dz}$)',
        #     xlim=(0.2, 1.2), ylim=(-0.3, 7.05), xscale='linear', yscale='linear')


class StudiesInfoTable(ParamTable):  # shot noise, effective z, ell limits, per zbin
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)


class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022JCAP...02..007W
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'],} # photometric redshift subsmaple
    info = {}

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        for k, v in StudiesInfoTable().getparams(zbin=self.zbin).to_dict().items(): setattr(self, k, v)


class TargetData(BaseTargetData, Studies.White2022):  # DESI LS DR9 LRGs correlated with Planck CMB Lensing
    path = f"{datapath}/White2022"  # Path to data from zenodo.org/records/5834378
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'], # photometric redshift subsmaple
            }
    
    info = {
        'area': 18000 *u.deg**2,  # area of DESI survey, [deg^2], Figure 1
        # info, Table 1
        'zMean': {# mean redshift
            'z1':0.47, 'z2':0.63, 'z3':0.79, 'z4':0.92},
        'ndens': {'units':u.deg**-2, # galaxy number desnity
            'z1':83, 'z2':149, 'z3':162, 'z4':149},
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
            
    def make_zdists(self, zbin=None, dz=None, zMin=None, zMax=None):
        self.require(['zbin'])
        self.z_df, self.nz_df = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_dndz.txt").T  # density distribution [deg^{-2}]
        
        zmin = zMin if zMin is not None else self.z_df.min()
        zmax = zMax if zMax is not None else self.z_df.max()
        self.dz = dz if dz is not None else self.z_df[1]-self.z_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)
        self.n_z = np.interp(self.z, self.z_df, self.nz_df) /u.deg**2
        self.N_z = self.n_z*self.area
        self.dNdz = self.N_z/self.dz
        self.dndz = self.n_z/self.dz
        self.dlnNdz = np.log(self.N_z.value)/self.dz


class Measurements(BaseMeasurement, Studies.White2022):  # DESI LS DR9 LRGs correlated with Planck CMB Lensing (White+ 2022, arxiv.org/abs/2111.09898)
    path = f"{datapath}/White2022"  # Path to data from zenodo.org/records/5834378
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'], # photometric redshift subsmaple
            }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)

        self.require(['zbin'])
            
        self.get_meas()

    def get_meas(self):
        self.ell, self.Cgg_data = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_cls.txt").T[0:2]  # ells and measured angular auto-spectra
        self.ell_model, self.Cgg_data_model = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_mod.txt").T[0:2]  # smooth model used to calculate cov mat
        
        self.C_covcomb = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_cov.txt")  # combined covariance matrix of auto and cross
        self.Cgg_cov = self.C_covcomb[:int(self.ell.size)]  # TODO: lazely assuming i can do this
        self.Cgg_err = np.diag(self.Cgg_cov)**0.5
        
        # weights of all multipoles used to get effective multipole ell, ell_effective = np.sum(weights*ells)
        self.auto_windowmat = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_wla.txt")  # window function matrix for auto-spec
        self.cross_windowmat = np.loadtxt(f"{self.path}/lrg_s0{self.zbin[-1]}_wlx.txt")


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2022JCAP...02..007W
    def Fig2(self, width=8, height=4):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$d \text{ln} N/dz$ (actually $\frac{1}{N}\frac{dN}{dz}$)',
            xlim=(0.2, 1.2), ylim=(-0.3, 7.05), xscale='linear', yscale='linear')
