"""
Measurements of the thermal Sunyaev-Zel'dovich effect with ACT and DESI luminous red galaxies


ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
arxiv.org/pdf/2502.08850
"""


from config import *
from Models.Papers.PlotsTables import ParamTable, read_wide_table
thispath = os.path.dirname(os.path.abspath(__file__))


    # def Fig2(self, width=6, height=5):
    #     return self.plot(filename='Fig2', width=width, height=height,
    #         xlabel=r'$z$', ylabel=r'$dN/dz$ (actually $n(z)$)',
    #         xlim=(0.1, 1.3), ylim=(0, 10.25), xscale='linear', yscale='linear')
        
    # def Fig3(self, width=12, height=10):
    #     return self.plot(filename=['Fig3a', 'Fig3b', 'Fig3c', 'Fig3d'], nrow=2, ncol=2, width=width, height=height,
    #         xlabel=r'$R [\text{arcmin}]$', ylabel=r'Compton Y-parameter [arcmin$^2$]',
    #         xlim=(0.75, 6.25), ylim=[(-0.8e-6, 4.4e-6), (-0.8e-6, 4.4e-6), (-1.15e-6, 4.2e-6), (-1.15e-6, 4.2e-6)], xscale='linear', yscale='linear')
        
    # def Fig4(self, width=12, height=10):
    #     return self.plot(filename=['Fig4a', 'Fig4b', 'Fig4c', 'Fig4d'], nrow=2, ncol=2, width=width, height=height,
    #         xlabel=r'$R [\text{arcmin}]$', ylabel=r'Compton Y-parameter [arcmin$^2$]',
    #         xlim=(0.75, 6.25), ylim=[(-0.45e-6, 5.5e-6), (-0.45e-6, 5.5e-6), (-0.5e-6, 5.5e-6), (-0.5e-6, 5.5e-6)], xscale='linear', yscale='linear')


class Studies(BaseStudy):  # Measurements of the thermal Sunyaev-Zel'dovich effect with ACT and DESI luminous red galaxies, ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subs = {
    }
    info = {
    }


class TargetDataInfoTable(ParamTable):  # rough mean halo mass (Yuan 2023) + T1, per zbin
    def __init__(self, filename=f"{thispath}/target_data_info.csv"):
        self.df = read_wide_table(filename)


class TargetData(BaseTargetData, Studies.Liu2025):  # ACT DR6 (&DR5) maps stacked on DESI LS DR9 LRGs
    path = f"{datapath}/Liu2025"  # path to data from zenodo.org/records/14706729
    subs = {
        'zbin' : ['all', 'z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        # 'ACTDR' : ['DR5', 'DR6'],  # ACT y map DR
        # 'freq' : ['090', '150'],  # y map frequency (DR5)
    }

    info = {
        'area': 7326 *u.deg**2,  # area of ACT DESI overlap, Fig 1
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

        if self.zbin == 'all':
            row = {k: None for k in ['logMhMean', 'zMean', 'nGal', 'NGal_unmasked', 'NGal']}
        else:
            row = TargetDataInfoTable().getparams(zbin=self.zbin).to_dict()
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v): v = None
            if k == 'nGal' and v is not None: v = v *u.deg**-2
            setattr(self, k, v)

        # Loading redshift distribution from file
        zcols = pd.read_csv(f"{datapath}/Zhou2023B/main_lrg_pz_dndz_iron_v0.4_dz_0.02.txt", sep=" ", nrows=1).columns[1:]  # get col names from Zhou2023
        self.zdf = pd.DataFrame(np.genfromtxt(f"{self.path}/fig2_main_lrg_pz_dndz_iron_v0.4_dz_0.01.txt"), columns=zcols)  # Spectroscopic distributions of four sub-sample photometric redshift bins
        self.zs_df = (self.zdf.zmin+self.zdf.zmax).values/2  # define zs at center of bins

    # def get_beam(self):
    #     self.require(['ACTDR'])
    #     if self.ACTDR=='DR6': 
    #         ACTDR6 = Coulton2024()
    #         self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data
    #     elif self.ACTDR=='DR5': 
    #         self.require(['freq'])
    #         ACTDR5 = Naess2020({'freq':self.freq})
    #         self.beam_ells, self.beam_data = ACTDR5.beam_ells, ACTDR5.beam_data
    #         self.resp_ells, self.resp_data = ACTDR5.resp_ells, ACTDR5.resp_data

    def make_zdist(self, dz=None, zMin=None, zMax=None, **kwargs):
        self.require(['zbin'])
        zstr = f'bin_{self.zbin[-1]}' if self.zbin!='all' else 'all'
        self.n_df = self.zdf[f'{zstr}_combined'].values /u.deg**2
        self.dndz_df = self.n_df /(self.zs_df[1]-self.zs_df[0])
        zmin = zMin if zMin is not None else self.zs_df.min()
        zmax = zMax if zMax is not None else self.zs_df.max()
        self.dz = dz if dz is not None else self.zs_df[1]-self.zs_df[0]
        self.z = np.arange(zmin, zmax+self.dz, self.dz)
        self.dndz = np.interp(self.z, self.zs_df, self.dndz_df)
        
        self.dNdz = self.dndz*self.area
        self.n_z = self.dndz*self.dz
        self.N_z = self.dNdz*self.dz

    def make_Mhdist(self, logMhMin=None, logMhMax=None, dlogMh=None, logMhNum=None, halomodel=None):
        self.logMhMin = 12
        self.logMhMax = 14
        self.catdist('logMh', np.zeros([100]), qMin=logMhMin, qMax=logMhMax, dq=dlogMh, qNum=logMhNum, densspace=1)


class Measurements(BaseMeasurement, Studies.Liu2025):  # ACT DR6 (&DR5) maps stacked on DESI LS DR9 LRGs (Liu+ 2025, arxiv.org/abs/2502.08850)
    path = f"{datapath}/Liu2025"  # path to data from zenodo.org/records/14706729
    sharedpath = f"{datapath}/Liu2025_shared"  # path to data from author
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'ACTDR' : ['DR5', 'DR6'],  # ACT y map DR
        'dp' :['beta', 'dBeta'],  # CIB deprojection method (DR6)
        'Beta' : ['fiducial', '1.2', '1.4', '1.6'],  # spectral index value in CIB deprojection (DR6)
        'TCIB' : ['10.7', '24.0'],  # TCIB value in CIB dprojection (DR6)
        'aper' : ['CAP', 'RingRing'],  # aperture used (DR5)
        'freq' : ['090', '150'],  # y map frequency (DR5)
    }

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['zbin', 'ACTDR'])

        if self.ACTDR=='DR6': 
            self.require(['zbin', 'dp', 'Beta', 'TCIB'])
        elif self.ACTDR=='DR5': 
            self.require(['zbin', 'aper', 'freq'])
            
        try: self.get_meas_shared()
        except: pass

        self.get_meas()

    def get_meas(self):
        if self.ACTDR=='DR6':  # ACT DR6 stacked y profiles, fiducial and using deprojection method/values:
            if self.dp=='beta' and self.TCIB=='10.7': y = pd.read_csv(f"{self.path}/fig3.csv")  # using CIB deprojected y maps
            elif self.dp=='dBeta' and self.TCIB=='10.7': y = pd.read_csv(f"{self.path}/fig4.csv")  # using CIB & dBeta moment deprojected y maps
            elif self.dp=='beta' and self.TCIB=='24.0': y = pd.read_csv(f"{self.path}/fig10.csv")  # using CIB deprojected y maps, T=24.0K
            elif self.dp=='dBeta' and self.TCIB=='24.0': y = pd.read_csv(f"{self.path}/fig11.csv")  # using CIB & dBeta moment deprojected y maps, T=24.0K
            
            dBetastr = f"Beta_{self.Beta}" if self.Beta!='fiducial' else 'fiducial'
            
            self.R = y['RApArcmin'].values[:-1] *u.arcmin
            self.tSZ_data= y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}"].values[:-1] *u.arcmin**2
            self.tSZ_err = y[f"pz{self.zbin[-1]}_act_dr6_{dBetastr}_err"].values[:-1] *u.arcmin**2

        elif self.ACTDR=='DR5':  # ACT DR5 stacked y profiles
            if self.aper=='CAP': y = pd.read_csv(f"{self.path}/fig12.csv")  # using standard CAP
            elif self.aper=='RingRing': y = pd.read_csv(f"{self.path}/fig13.csv")  # using ring-ring filter
            
            self.R = y['RApArcmin'].values *u.arcmin
            self.tSZ_data = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}"].values *u.arcmin**2
            self.tSZ_err = y[f"pz{self.zbin[-1]}_act_dr5_f{int(self.freq)}_err"].values *u.arcmin**2
            
    def get_meas_shared(self):
        if self.ACTDR=='DR6' and self.TCIB=='10.7':
            dBetastr = f"dBeta_{self.Beta}_{self.TCIB}" if self.Beta!='fiducial' else 'fiducial'
            y = np.genfromtxt(f"{self.sharedpath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/diskring_tsz_uniformweight_measured.txt").T
            ycov = np.genfromtxt(f"{self.sharedpath}/DESI_pz{self.zbin[-1]}_act_dr6_{dBetastr}/cov_diskring_tsz_uniformweight_bootstrap.txt").T

            self.R = y[0] *u.arcmin
            self.tSZ_data = (y[1] *u.sr).to(u.arcmin**2)
            self.tSZ_err = (y[2] *u.sr).to(u.arcmin**2)
            self.tSZ_cov = (ycov *u.sr**2).to(u.arcmin**4)


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    def Fig2(self, width=6, height=5):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$dN/dz$ (actually $n(z)$)',
            xlim=(0.1, 1.3), ylim=(0, 10.25), xscale='linear', yscale='linear')
        
    def Fig3(self, width=12, height=10):
        return self.plot(filename=['Fig3a', 'Fig3b', 'Fig3c', 'Fig3d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'$R [\text{arcmin}]$', ylabel=r'Compton Y-parameter [arcmin$^2$]',
            xlim=(0.75, 6.25), ylim=[(-0.8e-6, 4.4e-6), (-0.8e-6, 4.4e-6), (-1.15e-6, 4.2e-6), (-1.15e-6, 4.2e-6)], xscale='linear', yscale='linear')
        
    def Fig4(self, width=12, height=10):
        return self.plot(filename=['Fig4a', 'Fig4b', 'Fig4c', 'Fig4d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'$R [\text{arcmin}]$', ylabel=r'Compton Y-parameter [arcmin$^2$]',
            xlim=(0.75, 6.25), ylim=[(-0.45e-6, 5.5e-6), (-0.45e-6, 5.5e-6), (-0.5e-6, 5.5e-6), (-0.5e-6, 5.5e-6)], xscale='linear', yscale='linear')
