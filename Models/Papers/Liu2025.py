"""
Measurements of the thermal Sunyaev-Zel'dovich effect with ACT and DESI luminous red galaxies

ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
arxiv.org/pdf/2502.08850
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Liu2025")

datapath = f"{DATA_PATH}/Liu2025"  # path to data from zenodo.org/records/14706729
shareddatapath = f"{DATA_PATH}/Liu2025_shared"  # path to data from author


class TargetData():  # ACT DR6 (&DR5) maps stacked on DESI LS DR9 LRGs    
    area = 7326 *u.deg**2  # area of ACT DESI overlap, Fig 1

    UNITS = {'nGal': u.deg**-2}
    INTS = ['NGal_unmasked', 'NGal']

    def __init__(self, zbin, **kwargs):
        self.__dict__.update(locals())

        table1 = Table1()
        if zbin != 'all' and zbin not in table1.df.index:
            raise ValueError(f"zbin must be 'all' or one of {list(table1.df.index)}, got {zbin!r}")

        if self.zbin != 'all':
            row = table1.getparams(zbin=self.zbin).to_dict()
            for k, v in row.items():
                if k in self.INTS: v = int(v)
                if k in self.UNITS: v = v*self.UNITS[k]
                setattr(self, k, v)

        self.load_zdist()

    def load_zdist(self):
        # Loading redshift distribution from file
        zcols = pd.read_csv(f"{DATA_PATH}/Zhou2023B/main_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # get col names from Zhou2023
        self.zdf = pd.DataFrame(np.genfromtxt(f"{datapath}/fig2_main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"), columns=zcols)  # Spectroscopic distributions of four sub-sample photometric redshift bins
        self.zs_df = (self.zdf.zmin+self.zdf.zmax).values/2  # define zs at center of bins
        
        zstr = f'bin_{self.zbin[-1]}' if self.zbin!='all' else 'all'
        self.n_df = self.zdf[f'{zstr}_combined'].values /u.deg**2
        self.dndz_df = self.n_df /(self.zs_df[1]-self.zs_df[0])

    def make_zbins(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        zMin = zMin if zMin is not None else getattr(self, 'zMin', self.zs_df.min())
        zMax = zMax if zMax is not None else getattr(self, 'zMax', self.zs_df.max())
        zNum = zNum if zNum is not None else np.int32((zMax-zMin)/dz)
        
        zbins = np.linspace(zMin, zMax, zNum)
        z = (zbins[1:]+zbins[:-1])/2
        dz = z[1]-z[0]
        return zbins, z, dz
    
    def dndz(self, zMin=None, zMax=None, dz=None, zNum=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, dz=dz, zNum=zNum)
        return z, np.interp(z, self.zs_df, self.dndz_df)
    
    def dNdz(self, zMin=None, zMax=None, dz=None, zNum=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, dz=dz, zNum=zNum)
        return z, np.interp(z, self.zs_df, self.dndz_df)*self.area
    
    def N_z(self, zMin=None, zMax=None, dz=None, zNum=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, dz=dz, zNum=zNum)
        return z, np.interp(z, self.zs_df, self.dndz_df)*self.area*dz
    
    def n_z(self, zMin=None, zMax=None, dz=None, zNum=None, **kwargs):
        zbins, z, dz = self.make_zbins(zMin=zMin, zMax=zMax, dz=dz, zNum=zNum)
        return z, np.interp(z, self.zs_df, self.dndz_df)*dz
    
    
class Measurement:  
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'ACTDR' : ['DR5', 'DR6'],  # ACT y map DR
        'dp' :['beta', 'dBeta'],  # CIB deprojection method (DR6)
        'Beta' : ['fiducial', '1.2', '1.4', '1.6'],  # spectral index value in CIB deprojection (DR6)
        'TCIB' : ['10.7', '24.0'],  # TCIB value in CIB dprojection (DR6)
        'aper' : ['CAP', 'RingRing'],  # aperture used (DR5)
        'freq' : ['090', '150'],  # y map frequency (DR5)
    }

    def __init__(self, zbin, ACTDR, dp=None, Beta=None, TCIB=None, aper=None, freq=None, **kwargs):
        self.__dict__.update(locals())
        
        self.get_meas()
            
        try: self.get_meas_shared()
        except: pass

    def get_meas(self):
        if self.ACTDR=='DR6':  # ACT DR6 stacked y profiles, fiducial and using deprojection method/values:
            if self.dp=='beta' and self.TCIB=='10.7': y = pd.read_csv(f"{datapath}/fig3.csv")  # using CIB deprojected y maps
            elif self.dp=='dBeta' and self.TCIB=='10.7': y = pd.read_csv(f"{datapath}/fig4.csv")  # using CIB & dBeta moment deprojected y maps
            elif self.dp=='beta' and self.TCIB=='24.0': y = pd.read_csv(f"{datapath}/fig10.csv")  # using CIB deprojected y maps, T=24.0K
            elif self.dp=='dBeta' and self.TCIB=='24.0': y = pd.read_csv(f"{datapath}/fig11.csv")  # using CIB & dBeta moment deprojected y maps, T=24.0K
            
            dBetastr = f"Beta_{self.Beta}" if self.Beta!='fiducial' else 'fiducial'
            
            self.R = y['RApArcmin'].values[:-1] *u.arcmin
            self.y_data= y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}"].values[:-1] *u.arcmin**2
            self.y_err = y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}_err"].values[:-1] *u.arcmin**2

        elif self.ACTDR=='DR5':  # ACT DR5 stacked y profiles
            if self.aper=='CAP': y = pd.read_csv(f"{datapath}/fig12.csv")  # using standard CAP
            elif self.aper=='RingRing': y = pd.read_csv(f"{datapath}/fig13.csv")  # using ring-ring filter
            
            self.R = y['RApArcmin'].values *u.arcmin
            self.y_data = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}"].values *u.arcmin**2
            self.y_err = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}_err"].values *u.arcmin**2
            
    def get_meas_shared(self):
        if self.ACTDR=='DR6' and self.TCIB=='10.7':
            dBetastr = f"dBeta_{self.Beta}_{self.TCIB}" if self.Beta!='fiducial' else 'fiducial'
            y = np.genfromtxt(f"{shareddatapath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/diskring_tsz_uniformweight_measured.txt").T
            ycov = np.genfromtxt(f"{shareddatapath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/cov_diskring_tsz_uniformweight_bootstrap.txt").T

            self.R = y[0] *u.arcmin
            self.y_data = (y[1] *u.sr).to(u.arcmin**2)
            self.y_err = (y[2] *u.sr).to(u.arcmin**2)
            self.y_cov = (ycov *u.sr**2).to(u.arcmin**4)


# TABLE I. Mean redshift, mean number density, and number of objects (overall and after masking) per photometric redshift bin.
# logMhMean is not from Table I: it's the rough mean halo mass taken from Yuan 2023, III.A.p5
class Table1(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1.csv"):
        self.df = read_wide_table(filename)




class Fig2(BasePlots2):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(6, 5),
             xlabel=r'$z$', xlim=(0.1, 1.3), xscale='linear',
             ylabel=r'$dN/dz$ (actually $n(z)$)', ylim=(0, 10.25), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig3(BasePlots2):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subplots = [[
        dict(name='Fig3a', filename='Fig3a', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.8e-6, 4.4e-6), yscale='linear'),
        dict(name='Fig3b', filename='Fig3b', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.8e-6, 4.4e-6), yscale='linear'),
    ], [
        dict(name='Fig3c', filename='Fig3c', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-1.15e-6, 4.2e-6), yscale='linear'),
        dict(name='Fig3d', filename='Fig3d', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-1.15e-6, 4.2e-6), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig4(BasePlots2):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subplots = [[
        dict(name='Fig4a', filename='Fig4a', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.45e-6, 5.5e-6), yscale='linear'),
        dict(name='Fig4b', filename='Fig4b', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.45e-6, 5.5e-6), yscale='linear'),
    ], [
        dict(name='Fig4c', filename='Fig4c', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.5e-6, 5.5e-6), yscale='linear'),
        dict(name='Fig4d', filename='Fig4d', figsize=(6, 5),
             xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
             ylabel=r'Compton Y-parameter [arcmin$^2$]', ylim=(-0.5e-6, 5.5e-6), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)




    
    

            
            
            

