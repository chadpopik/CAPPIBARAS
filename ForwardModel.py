from config import *
    

# Base Model
class Cosmology():
    # Get all paper.py and code.py files where we'll look for our models            
    papers = [paper[:-3] for paper in os.listdir(f"{CAPPIBARAS_PATH}/Models/Papers") if paper.endswith(".py")]
    codes = [code[:-3] for code in os.listdir(f"{CAPPIBARAS_PATH}/Models/Codes") if code.endswith(".py")]
    # List of submodels used for a foward model, in the order they need to be loaded in
    submodels = ['Cosmology', 'HaloModel']
    
    def __init__(self, Models={}, VERBOSE=False, PAPERCHECK=False, TESTMODELS=False, **kwargs):
        
        # Auto-assign every constructor argument as an attribute of the same name.
        self.__dict__.update(locals())
        
        # Load in each of the submodels defined
        for submodel in [s for s in self.submodels if s in Models]:
            name = Models[submodel]['name']
            args = {k: v for k, v in Models[submodel].items() if k != 'name'}
            
            # Search for the model in Papers and Codes folder
            if name in self.papers: path = "Models.Papers"
            elif name in self.codes: path = "Models.Codes"
            else: raise ValueError(f"Submodel {name} not found in either Models/Codes or Models/Papers")
            
            # Load in the module and reload it to make sure we have the latest version
            if self.VERBOSE: print(f"\nImporting {name} {submodel} from {path}")
            module = importlib.reload(importlib.import_module(f"{path}.{name}"))
            
            # Now run the setup function for each of those submodels
            if self.VERBOSE: print(f"Initializing with {args}")
            getattr(self, f"setup_{submodel.lower()}")(getattr(module, submodel), **Models[submodel])
            
            
    """Cosmology class contains cosmological parameters and basic comsological functions"""
    def setup_cosmology(self, Cosmology, **cosmo_args):
        # Initialized with cosmological parameters, and a model for Plin (if needed)
        self.cosmology = Cosmology(**cosmo_args)
        
        # Functions used are H(z), dA(z), rhoc(z), Plin(k;z), etc.
        # Also use parameters like H0, Omega_M, omega_b, etc.
        # Instead of defining each one, we'll keep them all in self.cosmology
        
    """Halo Model class contains more complicated halo functions"""
    def setup_halomodel(self, HaloModel, **halomodel_args):
        # Initialized with cosmological parameters, and models for aspects of the halo model (mass function, bias, concentration, etc.)
        self.halomodel = HaloModel(**(self.Models['Cosmology'] | halomodel_args))  # Import Halo Model
        
        # Function used are dndM, bh, NFW, logM_conv, etc.
        # Instead of defining each one, we'll keep them all in self.halomodel
            
            
            
class Target_Distributions(Cosmology):
    # List of submodels used for a foward model, in the order they need to be loaded in
    submodels = Cosmology.submodels + ['SHMR', 'TargetData']
    
    def __init__(self, 
                #  zNum=10, dz=None, zMin=None, zMax=None,
                #  logMhNum=20, dlogMh=None, logMhMin=None, logMhMax=None,
                #  logMsNum=20, dlogMs=None, logMsMin=None, logMsMax=None,
                 **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        
    
    """Step 4: Setup target data, which should contain galaxy catalogs or predefined distributions for redshift and halo mass.
    We need this to get redshift and halo mass distributions for the next two steps.
    We load in specific arguments to specify which subsample we're using"""
    def setup_targetdata(self, TargetData, **targetdata_args):
        self.targetdata = TargetData(**targetdata_args)
        
        self.setup_zdistribution(**targetdata_args)
        
        self.v_rms = getattr(self.targetdata, "v_rms", None)
        
        self.setup_Mdistribution(**targetdata_args)

    
    
    """Step 4a: Setup redshift distribution
    We need this set up to estimate our effective z range, average z values for the projection, and taking a redshift average.
    We load in specific arguments from TargetData that specify things like zMin, zMax, dz, or zNum, which are used to construct the redshift distribution. We also load in cosmology for possible unit conversion, but it probably doesn't need it."""
    def setup_zdistribution(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        if self.PAPERCHECK: # Get preset values from the paper
            zMean_paper = getattr(self.targetdata, 'zMean', np.nan)
            zMin_paper = getattr(self.targetdata, 'zMin', np.nan)
            zMax_paper = getattr(self.targetdata, 'zMax', np.nan)
            print(f"Paper: {zMin_paper:.2f} < z < {zMax_paper:.2f}, zMean = {zMean_paper:.2f}")
            
        try:
            # Construct redshift distribution
            self.z, self.dNdz = self.targetdata.dNdz(zMin=zMin, zMax=zMax, zNum=zNum, dz=dz, **kwargs)
            
            # Take the mean value of the redshift distribution, which is used for the projection.
            self.zMean = np.trapezoid(self.z*self.dNdz, self.z)/np.trapezoid(self.dNdz, self.z)
        except:
            self.z = np.linspace(zMin, zMax, zNum)
            self.zMean = zMean_paper
        
        if self.VERBOSE:
            print(f"Model: {self.z.min():.2f} < z < {self.z.max():.2f}, zMean = {self.zMean:.2f}, dz = {(self.z[1]-self.z[0]):.2f}")

    
    """Step 4b: Setup halo mass distribution.
    We need this set up to estimate our effective halo mass range, and taking distribution weighted halo mass averages.
    We load in specific arguments from TargetData that specify things like logMhMin, logMhMax, dlogMh, or logMhNum, which are used to construct the halo mass distribution. We also load in cosmology, which is used to calculate density units for the halo mass distribution."""
    def setup_Msdistribution(self, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, dlogMh=None, logMhNum=None, **kwargs):
        logMsNum = logMsNum if logMsNum is not None else logMhNum
        dlogMs = dlogMs if dlogMs is not None else dlogMh
    
        self.logMs, self.dNdlogMs = self.targetdata.dNdlogMs(logMsMin=logMsMin, logMsMax=logMsMax, logMsNum=logMsNum, dlogMs=dlogMs, cosmology=self.cosmology, **kwargs)
        
        # Get mean value of halo mass, mostly just for comparison
        self.logMsMean = np.trapezoid(self.logMs*self.dNdlogMs, self.logMs)/np.trapezoid(self.dNdlogMs, self.logMs)
        
        if self.PAPERCHECK: # Get preset values from the paper
            logMsMean_paper = getattr(self.targetdata, 'logMsMean', np.nan)
            logMsMin_paper = getattr(self.targetdata, 'logMsMin', np.nan)
            logMsMax_paper = getattr(self.targetdata, 'logMsMax', np.nan)
            print(f"Paper: {logMsMin_paper:.2f} < logMs < {logMsMax_paper:.2f}, logMsMean = {logMsMean_paper:.2f}")
            

        if self.VERBOSE:
            print(f"Model: {self.logMs.min():.2f} < logMs < {self.logMs.max():.2f}, logMsMean = {self.logMsMean:.2f}, dlogMs = {(self.logMs[1]-self.logMs[0]):.2f}")
                


    """Step 4b: Setup halo mass distribution.
    We need this set up to estimate our effective halo mass range, and taking distribution weighted halo mass averages.
    We load in specific arguments from TargetData that specify things like logMhMin, logMhMax, dlogMh, or logMhNum, which are used to construct the halo mass distribution. We also load in cosmology, which is used to calculate density units for the halo mass distribution."""
    def setup_Mhdistribution(self, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, papcheck=True, **kwargs):
        if self.PAPERCHECK and papcheck: # Get preset values from the paper
            logMhMean_paper = getattr(self.targetdata, 'logMhMean', np.nan)
            logMhMin_paper = getattr(self.targetdata, 'logMhMin', np.nan)
            logMhMax_paper = getattr(self.targetdata, 'logMhMax', np.nan)
            print(f"Paper: {logMhMin_paper:.2f} < logMh < {logMhMax_paper:.2f}, logMhMean = {logMhMean_paper:.2f}")

        try:
            self.logMh, self.dNdlogMh = self.targetdata.dNdlogMh(logMhMin=logMhMin, logMhMax=logMhMax, logMhNum=logMhNum, dlogMh=dlogMh, cosmology=self.cosmology, **kwargs)
            
            if self.logMh.max()>16:
                self.logMh, self.dNdlogMh = self.targetdata.dNdlogMh(logMhMin=logMhMin, logMhMax=logMhMax, logMhNum=logMhNum, dlogMh=dlogMh, cosmology=self.cosmology, **kwargs)
            
            # Get mean value of halo mass, mostly just for comparison
            self.logMhMean = np.trapezoid(self.logMh*self.dNdlogMh, self.logMh)/np.trapezoid(self.dNdlogMh, self.logMh)
        
        except:            
            self.logMh = np.linspace(logMhMin, logMhMax, logMhNum)
            if self.logMh.max()>16:
                self.logMh = np.linspace(logMhMin, 16, logMhNum)
            self.logMhMean = logMhMean_paper
        
        if self.VERBOSE:
            print(f"Model: {self.logMh.min():.2f} < logMh < {self.logMh.max():.2f}, logMhMean = {self.logMhMean:.2f}, dlogMh = {self.logMh[1]-self.logMh[0]:.2f}")
            
    def setup_Mdistribution(self, **targetdata_args):
        try:
            self.setup_Msdistribution(**targetdata_args)
        except:
            pass
        
        try:
            self.setup_Mhdistribution(**targetdata_args)
        except:
            try:
                self.get_Mh_from_Ms()
                self.setup_Mhdistribution(papcheck=False, **targetdata_args)
            except:
                self.setup_Mhdistribution(papcheck=False, **targetdata_args)

    def setup_shmr(self, SHMR, **kwargs):
        # Just a test becasue of the way i did this
        SHMR(**kwargs,cosmology=self.cosmology,logMs = np.linspace(8, 12, 10))
        
        self.logMs_to_logMh = lambda logMs: SHMR(**kwargs,cosmology=self.cosmology,logMs = logMs).logMhalo()
        
    def get_Mh_from_Ms(self):
        logMhshmr_to_logMh = lambda logMhshmr: self.halomodel.logM_conv(z=self.zMean, logM=logMhshmr, MassDef='vir', newmdef='200c', Concentration=self.Models['SHMR']['Concentration'])[0]
        
        conv_func = lambda logMs: logMhshmr_to_logMh(self.logMs_to_logMh(logMs))
    
        self.targetdata.add_Mh_catalog(conversion=conv_func)
    

        

class Prof_to_Obs(Cosmology):
    # List of submodels used for a foward model, in the order they need to be loaded in
    submodels = Cosmology.submodels + ['MapData', 'CAP', 'BeamConvolution', 'Projection']
    
    def __init__(self, 
                 zMean=None,  # mean redshift of sample
                 R=None, # aperture radii to model to, if not set by a measurement
                 observable=None,  # observable, if not set by a measurement
                 beam=None,  # beam transfer function, if not set by mapdata
                 beam_ells=None, # corresponding beam ell values, if not set by mapdata
                 nu=None,  # needed if going to TtSZ, if not set by measurement
                 v_rms=None,  # needed if going to TkSZ, if not set by targetdata (later)
                 rNum=50,  # resolution of 3D radius array 
                 **kwargs):

        self.__dict__.update(locals())
        super().__init__(**kwargs)

    
    """Step 3: Setup map data, which should contain an effctive beam array and a corresponding array of ell values."""
    def setup_mapdata(self, MapData, **kwargs):
        # We load in specific arguments from MapData to specify the exact map beam we are trying to retrieve.
        self.mapdata = MapData(**kwargs)
        # We need for the map beams.
        self.beam, self.beam_ells = self.mapdata.beam_data, self.mapdata.beam_ells
        
        
    """Setup aperture photometry, which should contain function for converting a 2D beam-convoluted profile to a final measurement."""
    def setup_cap(self, CAP, **CAP_args):
        # Initialize with the apertures R to model to
        self.cap = CAP(**CAP_args, Rs=self.R)
        
        # Define the aperture photometry function we need
        self.aperture_photometry = self.cap.disk_ring_CAP
        
        # Maximum and minimum angular sizes require to perform the aperture photometry process above
        self.Rmax_CAP = self.cap.Rs_ring.max()
        self.Rmin_CAP = self.cap.Rs_disk.min()
        

    """Setup beam convolution, which should contain methods to take a projected 2D profile and convolve the beam and return back to a profile in R."""
    def setup_beamconvolution(self, BeamConv, **BeamConv_Args):
        # Initialize with the beam and beam ell arrays, and the angular size extent we'll need for the proceeding step
        self.beamconv = BeamConv(**BeamConv_Args, b_ell=self.beam_ells, bTF=self.beam, RMax=self.Rmax_CAP, RMin=self.Rmin_CAP)
        
        # Set the beam convolution function we need to convolve the 2D profile
        self.convolve_beam = self.beamconv.beam_convolve
        
        # Array of angular size values required to perform the beam convolution function above
        self.R_beam = self.beamconv.R_beam_unpad

        
    """Setup line of sight projection, which should contain methods to take a 3D profile and projection it along the line of sight to 2D."""
    def setup_projection(self, Projection, **proj_args):
        # Initialize with the angular radii needed to the proceeding step, and the mean angular distance of the sample
        self.projection = Projection(**proj_args, Rs_beam=self.R_beam, zMean=self.zMean, cosmology=self.cosmology)
        
        # Set projection function as attribute
        self.project_LOS = self.projection.proj2D

        # Use the projection's required extent to define a 3D radius array
        self.r = np.geomspace(self.projection.r3D.min(), self.projection.r3D.max(), self.rNum)
        
        self.profunit_to_obsunit = getattr(self.projection, f"to_{self.observable}", None)(nu=self.nu, v_rms=self.v_rms)
        
        if self.VERBOSE:
            print(f"Model: {self.r.min():.2e} < r < {self.r.max():.2f}")
            
    def prof_to_obs(self, prof):
        prof2d = self.project_LOS(self.r, prof)
        Rbeam, prof2d_beam = self.convolve_beam(prof2d)
        profCAP = self.aperture_photometry(Rbeam, prof2d_beam)
        prof_obs = self.profunit_to_obsunit * profCAP
        return prof_obs
            

            
class ProfileModel(Cosmology):
    # List of submodels used for a foward model, in the order they need to be loaded in
    submodels = Cosmology.submodels + ['Pressure', 'Density']
            
    def __init__(self, 
                 r=None,  # 3D radial distance array, in Mpc
                 z=None,  # redshift array
                 logMh=None,  # halo mass array, in log(Mh/Msun)
                 **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
            
    def setup_pressure(self, Pressure, **pressure_args):
        self.pressure = Pressure(r=self.r, z=self.z, logM200c=self.logMh, halomodel=self.halomodel, **pressure_args)
                        
        self.prof1h = self.pressure.onehalo
        self.prof2h = self.pressure.twohalo
        self.prof = self.pressure.total
        
        
    def setup_density(self, Density, **density_args):
        self.density = Density(r=self.r, z=self.z, logM200c=self.logMh, halomodel=self.halomodel, **density_args)

        self.prof1h = self.density.onehalo
        self.prof2h = self.density.twohalo
        self.prof = self.density.total
        



            
class Summary_Statistics(Cosmology):
    # List of submodels used for a foward model, in the order they need to be loaded in
    submodels = Cosmology.submodels + ['HOD', 'Spectra']
    
    def __init__(self, 
                 r=None, # 3D radial distance array, in Mpc
                 z=None, # redshift array
                 logMh=None, # halo mass array, in log(Mh/Msun)
                 dNdz=None, # redshift number distribution
                 dNdlogMh=None, # halo mass number distribution
                 **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        
    
    
    """Step 9: Setup HOD model, which should have functions for Ncen, Nsat, ncen, and nsat.
    We'll need this to take proper weighted integral for the averaging.
    We load in the arrays for radius, redshift, and mass, which won't change for the HOD, and then load in specific arguments that specify the HOD model, and our halomodel which we might need to calculate some things."""
    def setup_hod(self, HOD, **kwargs):
        self.logMvir = self.halomodel.logM_conv(z=self.z, logM=self.logMh, newmdef='vir')
        self.hod = HOD(r=self.r, z=self.z, logMh=self.logMvir, cosmology=self.cosmology, halomodel=self.halomodel, **kwargs)  # Import HOD Model
        self.Ncen, self.Nsat, self.ucen, self.usat = self.hod.Ncen, self.hod.Nsat, self.hod.ucen, self.hod.usat
        self.k = self.hod.k
        
        
    """Step 10: Setup HOD averaging process, which should be able to take a profile in r, z, Mh and take the halo mass average with the HOD factored in.
    We load in the arrays for radius, redshift, and mass, which won't change for the HOD, and then load in specific arguments that specify the HOD model, and our halomodel which we might need to calculate some things."""
    def setup_spectra(self, Spectra, **kwargs):
        self.spectra = Spectra(rs=self.r, z=self.z, logMh=self.logMh, Ncen=self.Ncen, Nsat=self.Nsat, ucen=self.ucen, usat=self.usat, dndlogMh=self.halomodel.dndlogm, **kwargs)  # Import Spectra/Averaging Model
        
        self.ave_hod = self.spectra.HODweighting
        
        self.setup_distaves()
        
        
    # Averages from distributions, if they're given
    def setup_distaves(self):
        try:
            dNdz_norm = self.dNdz/np.trapezoid(self.dNdz, self.z)
            self.ave_z = lambda prof: np.trapezoid(prof*dNdz_norm, self.z)
        except: 
            pass
        
        try:
            dNdlogm_norm = self.dNdlogMh/np.trapezoid(self.dNdlogMh, self.logMh)
            self.ave_Mh = lambda prof: np.trapezoid(prof*dNdlogm_norm, self.logMh)
        except:
            pass
        

    def average_prof(self, prof):
        prof_Mave = self.ave_hod(prof)
        # prof_Mave = self.ave_Mh(prof)
        
        prof_zMave = self.ave_z(prof_Mave)
        return prof_zMave



class SZObservable(Target_Distributions, Prof_to_Obs, ProfileModel, Summary_Statistics, Cosmology):   
    # List of submodels used for a foward model, in the order they need to be loaded in
    submodels = Cosmology.submodels + ['Measurement']+ Target_Distributions.submodels[2:] + Prof_to_Obs.submodels[2:] + Summary_Statistics.submodels[2:] + ProfileModel.submodels[2:]

    def __init__(self, 
                 **kwargs):
        self.__dict__.update(locals())
        
        super().__init__(**kwargs)
        
    """Step 2: Setup measurements, which should contain the aperture R values, some data array, and some error array or covariance matrix, but the latter two aren't strictly needed if you're not fitting."""
    def setup_measurement(self, Measurement, **kwargs):
        # We load in specific arguments from Measurement to specify the exact measuremnt we are trying to retrieve
        self.measurement = Measurement(**kwargs)
        # We also need this to calculate the array of r values needed to cover our the R aperture values
        self.R = self.measurement.R
        
        self.nu = getattr(self.measurement, "nu", None)
        
        # We need this to know what units/observable we're modeling the profile to.
        for meas in ['y', 'TkSZ', 'TtSZ']:
            for val in ['data', 'cov', 'err']:
                try:
                    setattr(self, f"meas_{val}", getattr(self.measurement, f"{meas}_{val}"))
                    self.observable = meas
                except:
                    pass

    
    def forward_model(self, params):
        return self.prof_to_obs(self.average_prof(self.prof(params)))
    

    def test_time(self, n_trials=100):
        steps = [
            "onehalo",
            "twohalo",
            "ave_hod",
            "ave_z",
            "project_LOS",
            "convolve_beam",
            "aperture_photometry",
            "unit_conversion",
        ]
        timings = {s: [] for s in steps}

        for _ in range(n_trials):
            t0 = time.perf_counter()
            prof1h = self.density.onehalo()
            t1 = time.perf_counter()

            prof2h = self.density.twohalo()
            t2 = time.perf_counter()

            prof_sum = prof1h + prof2h

            prof_Mave = self.ave_hod(prof_sum)
            # prof_Mave = self.ave_Mh(prof_sum)
            t3 = time.perf_counter()

            prof_zMave = self.ave_z(prof_Mave)
            t4 = time.perf_counter()

            prof2d = self.project_LOS(self.r, prof_zMave)
            t5 = time.perf_counter()

            Rbeam, prof2d_beam = self.convolve_beam(prof2d)
            t6 = time.perf_counter()

            profCAP = self.aperture_photometry(Rbeam, prof2d_beam)
            t7 = time.perf_counter()

            prof_obs = self.profunit_to_obsunit * profCAP
            t8 = time.perf_counter()

            timings["onehalo"].append(t1 - t0)
            timings["twohalo"].append(t2 - t1)
            timings["ave_hod"].append(t3 - t2)
            timings["ave_z"].append(t4 - t3)
            timings["project_LOS"].append(t5 - t4)
            timings["convolve_beam"].append(t6 - t5)
            timings["aperture_photometry"].append(t7 - t6)
            timings["unit_conversion"].append(t8 - t7)

        stats = {s: np.array(v) *1e6 for s, v in timings.items()}  # convert to us

        # print summary
        for s in steps:
            arr = stats[s]
            p10, median, p90 = np.percentile(arr, [10, 50, 90])
            print(f"{s:22s} mean={arr.mean():8.3f} us  std={arr.std():8.3f} us  "
                f"median={median:8.3f} us  p10={p10:8.3f} us  p90={p90:8.3f} us")

        return stats

    def plot_test_time(self, stats, bins=20, percentiles=(1, 99)):
        steps = list(stats.keys())

        n = len(steps)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, s in zip(axes, steps):
            arr = stats[s]
            lo, hi = np.percentile(arr, percentiles)
            ax.hist(arr, bins=bins, range=(lo, hi), color="steelblue", edgecolor="black", alpha=0.8)
            ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1,
                    label=f"mean={arr.mean():.2f} us")
            ax.axvline(np.median(arr), color="green", linestyle="--", linewidth=1,
                    label=f"median={np.median(arr):.2f} us")
            ax.set_title(s)
            ax.set_xlabel("time [us]")
            ax.set_ylabel("count")
            ax.legend(fontsize=8)

        # hide unused subplots
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle("Step timing distributions", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        



















# class SZObservable():    
#     # Get all paper.py and code.py files where we'll look for our models            
#     papers = [paper[:-3] for paper in os.listdir(f"{CAPPIBARAS_PATH}/Models/Papers") if paper.endswith(".py")]
#     codes = [code[:-3] for code in os.listdir(f"{CAPPIBARAS_PATH}/Models/Codes") if code.endswith(".py")]
    
#     # List of submodels used for a foward model, in the order they need to be loaded in
#     submodels = ['Cosmology', 'HaloModel', 'Measurement', 'TargetData', 'SHMR', 'MapData', 'CAP', 'BeamConvolution', 'Projection', 'Pressure', 'Density', 'HOD', 'Spectra']

#     def __init__(self, Models={}, 
#                  VERBOSE=False, PAPERCHECK=False, TESTMODELS=False, 
#                  rNum=50, zNum=20, logMhNum=30,
#                  zMean=None,R=None,z=None,logMh=None,r=None,k=None,beam_ells=None, beam=None, measunit=None,
#                  **kwargs):
        
#         # Auto-assign every constructor argument as an attribute of the same name.
#         self.__dict__.update(locals())
        
#         # Load in each of the submodels defined
#         for submodel in [s for s in self.submodels if s in Models]:
#             name = Models[submodel]['name']
#             args = {k: v for k, v in Models[submodel].items() if k != 'name'}
            
#             # Search for the model in Papers and Codes folder
#             if name in self.papers: path = "Models.Papers"
#             elif name in self.codes: path = "Models.Codes"
#             else: raise ValueError(f"Submodel {name} not found in either Models/Codes or Models/Papers")
            
#             # Load in the module and reload it to make sure we have the latest version
#             if self.VERBOSE: print(f"\nImporting {name} {submodel} from {path}")
#             module = importlib.reload(importlib.import_module(f"{path}.{name}"))
            
#             # Now run the setup function for each of those submodels
#             if self.VERBOSE: print(f"Initializing with {args}")
#             getattr(self, f"setup_{submodel.lower()}")(getattr(module, submodel), **Models[submodel])
            
            
#     def average_prof(self, prof):
#         prof_Mave = self.ave_hod(prof)
#         # prof_Mave = self.ave_smf(prof)
#         prof_zMave = self.ave_z(prof_Mave)
#         return prof_zMave
    
#     def prof_to_obs(self, prof):
#         prof2d = self.project_LOS(self.r, prof)
#         Rbeam, prof2d_beam = self.convolve_beam(prof2d)
#         profCAP = self.aperture_photometry(Rbeam, prof2d_beam)
#         prof_obs = self.measunit * profCAP
#         return prof_obs
    
#     def forward_model(self, params):
#         return self.prof_to_obs(self.average_prof(self.prof(params)))

        
#     """Cosmology class that contain functions like H(z), dA(z), rhoc(z), Plin(k;z), etc, as well as basic cosmological parameters such as H0, Omega_M, Omega_b, sigma8, ns, etc. """
#     def setup_cosmology(self, Cosmology, **kwargs):
#         # Arguments are typically cosmological parameters and possible a model for Plin
#         self.cosmology = Cosmology(**kwargs)
        
#         # We need this set up to do some volume calculations for unit conversions in dN/dz, dN/dM, and dA(z) used to estimate our required r array
#         # Instead of defining each one, we'll keep them all in self.cosmology
        
#     """Setup Halo Model
#     We'll need this to fetch more complicated models like NFW, dndM, Plin, bh, etc for calculating things in the HOD or Profiles.
#     We load in specific arguments that specify things of the halo model, and our set cosmological parameters to override the others."""
#     def setup_halomodel(self, HaloModel, **kwargs):
#         self.halomodel = HaloModel(**(self.Models['Cosmology'] | kwargs))  # Import Halo Model
                
            
#     """Step 2: Setup measurements, which should contain the aperture R values, some data array, and some error array or covariance matrix, but the latter two aren't strictly needed if you're not fitting."""
#     def setup_measurement(self, Measurement, **kwargs):
#         # We load in specific arguments from Measurement to specify the exact measuremnt we are trying to retrieve
#         self.measurement = Measurement(**kwargs)
#         # We need this to know what units/observable we're modeling the profile to.
#         self.setup_data()
#         # We also need this to calculate the array of r values needed to cover our the R aperture values
#         self.R = self.measurement.R
        
        
#     """Step 3: Setup map data, which should contain an effctive beam array and a corresponding array of ell values."""
#     def setup_mapdata(self, MapData, **kwargs):
#         # We load in specific arguments from MapData to specify the exact map beam we are trying to retrieve.
#         self.mapdata = MapData(**kwargs)
#         # We need for the map beams.
#         self.beam, self.beam_ells = self.mapdata.beam_data, self.mapdata.beam_ells
    
    
#     """Step 4: Setup target data, which should contain galaxy catalogs or predefined distributions for redshift and halo mass.
#     We need this to get redshift and halo mass distributions for the next two steps.
#     We load in specific arguments to specify which subsample we're using"""
#     def setup_targetdata(self, TargetData, **targetdata_args):
#         self.targetdata = TargetData(**targetdata_args)
        
#         self.setup_zdistribution()
#         # We try to set up the halo mass distribution, but if it fails then we try to set up the stellar mass distribution instead. This is because some papers only provide stellar mass distributions and not halo mass distributions, and we can convert between them using the SHMR.
#         try:
#             self.setup_Mhdistribution()
#         except: 
#             self.setup_Msdistribution()
#         # self.setup_distave()
#         self.setup_zdistave()
    
    
#     """Step 4a: Setup redshift distribution
#     We need this set up to estimate our effective z range, average z values for the projection, and taking a redshift average.
#     We load in specific arguments from TargetData that specify things like zMin, zMax, dz, or zNum, which are used to construct the redshift distribution. We also load in cosmology for possible unit conversion, but it probably doesn't need it."""
#     def setup_zdistribution(self, **kwargs):
#         if self.PAPERCHECK: # Get preset values from the paper
#             zMean_paper = getattr(self.targetdata, 'zMean', np.nan)
#             zMin_paper = getattr(self.targetdata, 'zMin', np.nan)
#             zMax_paper = getattr(self.targetdata, 'zMax', np.nan)
#             print(f"Paper: {zMin_paper:.2f} < z < {zMax_paper:.2f}, zMean = {zMean_paper:.2f}")
            
#         # Construct redshift distribution
#         self.z, self.dNdz = self.targetdata.dNdz(zNum=self.zNum)
        
#         # Take the mean value of the redshift distribution, which is used for the projection.
#         self.zMean = np.trapezoid(self.z * self.dNdz, self.z) / np.trapezoid(self.dNdz, self.z)
        
#         if self.VERBOSE:
#             print(f"Model: {self.z.min():.2f} < z < {self.z.max():.2f}, zMean = {self.zMean:.2f}, dz = {(self.z[1]-self.z[0]):.2f}")

    
#     """Step 4b: Setup halo mass distribution.
#     We need this set up to estimate our effective halo mass range, and taking distribution weighted halo mass averages.
#     We load in specific arguments from TargetData that specify things like logMhMin, logMhMax, dlogMh, or logMhNum, which are used to construct the halo mass distribution. We also load in cosmology, which is used to calculate density units for the halo mass distribution."""
#     def setup_Msdistribution(self, **kwargs):        
#         self.logMs, self.dndlogMs = self.targetdata.dndlogMs(cosmology=self.cosmology, logMsNum=getattr(self, 'logMsNum', getattr(self, 'logMhNum', None)))
                
#         # Get mean value of halo mass, mostly just for comparison
#         self.logMsMean = np.trapezoid(self.logMs * self.dndlogMs, self.logMs) / np.trapezoid(self.dndlogMs, self.logMs)
        
#         if self.PAPERCHECK: # Get preset values from the paper
#             logMsMean_paper = getattr(self.targetdata, 'logMsMean', np.nan)
#             logMsMin_paper = getattr(self.targetdata, 'logMsMin', np.nan)
#             logMsMax_paper = getattr(self.targetdata, 'logMsMax', np.nan)
#             print(f"Paper: {logMsMin_paper:.2f} < logMs < {logMsMax_paper:.2f}, logMsMean = {logMsMean_paper:.2f}")

#         if self.VERBOSE:
#             print(f"Model: {self.logMs.min():.2f} < logMs < {self.logMs.max():.2f}, logMsMean = {self.logMsMean:.2f}, dlogMs = {(self.logMs[1]-self.logMs[0]):.2f}")


#     """Step 4b: Setup halo mass distribution.
#     We need this set up to estimate our effective halo mass range, and taking distribution weighted halo mass averages.
#     We load in specific arguments from TargetData that specify things like logMhMin, logMhMax, dlogMh, or logMhNum, which are used to construct the halo mass distribution. We also load in cosmology, which is used to calculate density units for the halo mass distribution."""
#     def setup_Mhdistribution(self, **kwargs):
#         # Set here to prevent pyccl from breaking
#         if self.targetdata.cat_logMh.max()>=16: logMhMax=16
#         else: logMhMax=None
        
#         self.logMh, self.dndlogMh = self.targetdata.dndlogMh(cosmology=self.cosmology, logMhNum=self.logMhNum, logMhMax=logMhMax)
        
#         # Get mean value of halo mass, mostly just for comparison
#         self.logMhMean = np.trapezoid(self.logMh * self.dndlogMh, self.logMh) / np.trapezoid(self.dndlogMh, self.logMh)
        
#         if self.PAPERCHECK: # Get preset values from the paper
#             logMhMean_paper = getattr(self.targetdata, 'logMhMean', np.nan)
#             logMhMin_paper = getattr(self.targetdata, 'logMhMin', np.nan)
#             logMhMax_paper = getattr(self.targetdata, 'logMhMax', np.nan)
#             print(f"Paper: {logMhMin_paper:.2f} < logMh < {logMhMax_paper:.2f}, logMhMean = {logMhMean_paper:.2f}")
        
#         if self.VERBOSE:
#             print(f"Model: {self.logMh.min():.2f} < logMh < {self.logMh.max():.2f}, logMhMean = {self.logMhMean:.2f}, dlogMh = {self.logMh[1]-self.logMh[0]:.2f}")
        
#     def setup_shmr(self, SHMR, **kwargs):
#         conv_func = lambda logMs: self.halomodel.logM_conv(z=self.zMean, logM=SHMR(**kwargs,h=cosmology.h,logMs = logMs).logMhalo(), MassDef='vir', newmdef='200c', Concentration=kwargs['Concentration'])[0]
        
#         self.targetdata.add_Mh_catalog(conversion=conv_func)

#         self.setup_Mhdistribution()

#     """Step 4c: Setup weighted distribution average, which takes a 3D profile as a function of r, z, and logMh, and averages it using the distributions above.
#     We only need this if we're doing the average this way, if using the HOD average then we can skip it."""
#     def setup_distave(self):
#         dndzdlogmhalo_norm = self.dndlogMh*self.dNdz[:, None]/np.trapezoid(np.trapezoid(self.dndlogMh, self.logMh)*self.dNdz, self.z)
#         self.ave_smf = lambda prof: np.trapezoid(np.trapezoid(prof*dndzdlogmhalo_norm, self.logMh), self.z)
        
#     def setup_zdistave(self):
#         dndz_norm = self.dNdz/np.trapezoid(self.dNdz, self.z)
#         self.ave_z = lambda prof: np.trapezoid(prof*dndz_norm, self.z)

            
            
        
#     """Step 5: Setup aperture photometry, which should contain function for converting a 2D beam-convoluted profile to a final measurement."""
#     def setup_cap(self, CAP, **kwargs):
#         # We initilize this with specific arguments from CAP to specify the exact CAP process we're trying to recreate
#         self.cap = CAP(Rs=self.R, **kwargs)
#         # We need this to perform aperture photometry
#         self.aperture_photometry = self.cap.disk_ring_CAP
#         # We also need this to get the f_cap factor that informs our max radial extent.
#         self.Rmax_CAP = self.cap.Rs_ring.max()
#         self.Rmin_CAP = self.cap.Rs_disk.min()


#     """Step 6: Setup beam convolution, which should contain methods to take a projected 2D profile and convolve the beam and return back to a profile in R."""
#     def setup_beamconvolution(self, BeamConv, **kwargs):
#         # We also load in specific arguments that determine the specifics of the max angular size and resolution
        
#         self.beamconv = BeamConv(**kwargs,
#             # We  need the beam ells and profile ells
#             b_ell=self.beam_ells, bTF=self.beam,
#             # We need the angular size bounds of the CAP to determine the bounds of the harmonic Space where we'll do the convolution
#             RMax = self.Rmax_CAP, RMin = self.Rmin_CAP
#         )
#         self.convolve_beam = self.beamconv.beam_convolve
#         # We need the beam angular sizes to construct
#         self.R_beam = self.beamconv.R_beam_unpad

        
#     """Step 7: Setup line of sight projection, which should contain methods to take a 3D profile and projection it along the line of sight to 2D."""
#     def setup_projection(self, Projection, **kwargs):
#         self.projection = Projection(
#             # and the angular radii we'll need from the beam convolution
#             Rs_beam=self.R_beam, 
#             # we need to know the angular distance at the mean redshift to know our sky extent too, which we'll try to get from the redshift distribution but otherwise we can set it as a value in the input
#             AngDist=self.cosmology.dA(self.zMean), 
#             # We load in specific arguments that determine the specifics of the LOS max/min/resolution
#             **kwargs)
#         self.project_LOS = self.projection.proj2D
        
#         self.r = np.geomspace(self.projection.r3D.min(), self.projection.r3D.max(), self.rNum)
        
#         if self.VERBOSE:
#             print(f"Model: {self.r.min():.2e} < r < {self.r.max():.2f}")
            
        
        
        
        
    
    
    
#     """Step 9: Setup HOD model, which should have functions for Ncen, Nsat, ncen, and nsat.
#     We'll need this to take proper weighted integral for the averaging.
#     We load in the arrays for radius, redshift, and mass, which won't change for the HOD, and then load in specific arguments that specify the HOD model, and our halomodel which we might need to calculate some things."""
#     def setup_hod(self, HOD, **kwargs):
#         self.logMvir = self.halomodel.logM_conv(z=self.z, logM=self.logMh, newmdef='vir')
#         self.hod = HOD(r=self.r, z=self.z, logMh=self.logMvir, cosmology=self.cosmology, halomodel=self.halomodel, **kwargs)  # Import HOD Model
#         self.Ncen, self.Nsat, self.ucen, self.usat = self.hod.Ncen, self.hod.Nsat, self.hod.ucen, self.hod.usat
#         self.k = self.hod.k
        
        
#     """Step 10: Setup HOD averaging process, which should be able to take a profile in r, z, Mh and take the halo mass average with the HOD factored in.
#     We load in the arrays for radius, redshift, and mass, which won't change for the HOD, and then load in specific arguments that specify the HOD model, and our halomodel which we might need to calculate some things."""
#     def setup_spectra(self, Spectra, **kwargs):
#         self.spectra = Spectra(rs=self.r, z=self.z, logMh=self.logMh, Ncen=self.Ncen, Nsat=self.Nsat, ucen=self.ucen, usat=self.usat, dndlogMh=self.halomodel.dndlogm, **kwargs)  # Import Spectra/Averaging Model
#         self.ave_hod = self.spectra.HODweighting
    

#     def setup_pressure(self, Pressure, **kwargs):
#         self.pressure = Pressure(r=self.r, z=self.z, logM200c=self.logMh, halomodel=self.halomodel, **kwargs)
#         self.prof1h = self.pressure.onehalo
#         self.prof2h = self.pressure.twohalo_fixed
#         self.prof = lambda p={}: self.prof1h(p) + self.prof2h(p)

#         if hasattr(self, 'measurement'):
#             if hasattr(self.measurement, 'y_data'):
#                 self.measunit = self.density.to_y
#             elif hasattr(self.measurement, 'TtSZ_data'):
#                 self.measunit = self.density.to_TtSZ
        
#     def setup_density(self, Density, **kwargs):
#         self.density = Density(r=self.r, z=self.z, logM200c=self.logMh, halomodel=self.halomodel, **kwargs)
#         if hasattr(self, 'measurement'):
#             if hasattr(self.measurement, 'TkSZ_data'):
#                 self.measunit = self.density.to_TkSZ

#         self.prof1h = self.density.onehalo
#         self.prof2h = self.density.twohalo_fixed
#         self.prof = lambda p={}: self.prof1h(p) + self.prof2h(p)
        
#     def setup_data(self):
#         if hasattr(self.measurement, 'y_data'):
#             self.meas_data = self.measurement.y_data
#             self.meas_cov = self.measurement.y_cov
#             self.meas_err = self.measurement.y_err
            
#         elif hasattr(self.measurement, 'TkSZ_data'):
#             self.meas_data = self.measurement.TkSZ_data
#             self.meas_cov = self.measurement.TkSZ_cov
#             self.meas_err = self.measurement.TkSZ_err
            
#         elif hasattr(self.measurement, 'TtSZ_data'):
#             self.meas_data = self.measurement.TtSZ_data
#             self.meas_cov = self.measurement.TtSZ_cov
#             self.meas_err = self.measurement.TtSZ_err




# class TtSZ(SZObservable):
#     def setup_pressure(self, Pressure, **kwargs):
#         self.pressure = Pressure(r=self.r, z=self.z, logMh=self.logMh, halomodel=self.halomodel, **kwargs)
#         self.prof = self.pressure.total()
#         self.prof1h = self.pressure.onehalo()
#         self.prof2h = self.pressure.twohalo()
        
#         # self.measunit = pres_to_y(**self.cosmopars)
        
#     def setup_data(self):
#         self.meas_data = self.measurement.TtSZ_data
#         self.meas_cov = self.measurement.TtSZ_cov
        
        
# class Comptony(SZObservable):
#     def setup_pressure(self, Pressure, **kwargs):
#         self.pressure = Pressure(r=self.r, z=self.z, logMh=self.logMh, halomodel=self.halomodel, **kwargs)
#         self.prof = self.pressure.total
#         self.prof1h = self.pressure.onehalo
#         self.prof2h = self.pressure.twohalo
        
#         # self.measunit = pres_to_y(**self.cosmopars)

#     def setup_data(self):
#         self.meas_data = self.measurement.y_data
#         self.meas_cov = self.measurement.y_cov
#         self.meas_err = self.measurement.y_err


# class TkSZ(SZObservable):
#     def setup_density(self, Density, **kwargs):
#         self.density = Density(r=self.r, z=self.z, logM200c=self.logMh, halomodel=self.halomodel, **kwargs)
#         self.measunit = self.density.to_TkSZ

#         self.prof1h = self.density.onehalo
#         self.prof2h = self.density.twohalo_fixed
#         self.prof = lambda p={}: self.prof1h(p) + self.prof2h(p)

        
#     def setup_data(self):
#         self.meas_data = self.measurement.TkSZ_data
#         self.meas_cov = self.measurement.TkSZ_cov
#         self.meas_err = self.measurement.TkSZ_err
















# def pres_to_y(XH, **kwargs):  # factor to convert projected Pressure to compton-y
#     return (c.sigma_T/c.m_e/c.c**2).cgs * (2+2*XH)/(3+5*XH) * u.g/u.Msun * u.Msun.to(u.g)

# def y_to_uK(nu, T_CMB, **kwargs):  # factor to convert compton-y to uK
#     x = (c.h * nu / (c.k_B * T_CMB)).decompose().value
#     fnu = x / np.tanh(x / 2.0) - 4.0
#     return fnu*T_CMB*u.uK*1e6

# def pres_to_uK(nu, **kwargs):  # factor to convert projected Pressure to uK
#     return pres_to_y(**kwargs)*y_to_uK(nu, **kwargs)

# def rho_to_TkSZ(v_rms, XH, T_CMB, **kwargs):  # factor to convert projected density to uK
#     return v_rms * (c.sigma_T/c.m_p).cgs * (1+XH)/2 * T_CMB*u.uK*1e6 * u.cm.to(u.Mpc)**2 * u.Mpc**2/u.cm**2 * u.g/u.Msun * u.Msun.to(u.g)


# class ForwardModel:
#     def __init__(self,
#         # the following should all be uninitialized classes or already-built objects, e.g. Models.Codes.pyccl.Cosmology, Models.Papers.Liu2025.TargetData, etc.
#         Cosmology=None,  # handles cosmological parameters and calculations that only depend on them, e.g. H, dA, rhoc
#         TargetData=None,  # sample properties: builds the z and halo-mass distributions the profile gets averaged over
#         MapData=None,  # map info (beams, responses) used when projecting/convolving the 3D profile
#         HaloModel=None,  # halo properties needed by the profile: rhoc, dndlogm, bh, Plin

#         Projection=None,  # projects the 3D profile to a 2D beam-convolved aperture-photometered observable; needs Measurement. Skip to get just the mass/z-averaged 3D profile
#         Measurement=None,  # measurement data (R values, aperture); required if Projection is given, since Projection derives its r range and R values from it
#         HOD=None,  # HOD model for galaxy weighting; skip to average over the halo mass distribution instead
#         Spectra=None,  # HOD-weighting/averaging class (e.g. Models.Papers.Popik2026.Spectra); required if HOD is given

#         configs=None,
#         # single dict of per-submodel config kwargs, keyed by submodel name, e.g.
#         # {'TargetData': {'zbin': 'z1', ...}, 'HaloModel': {'MassFunc': 'Tinker08', ...}, 'Profile': {...}, ...}
#         # matching the per-submodel blocks of a SOLikelihoods2 yaml (the 'Profile' entry is used by
#         # setup_profile() in TSZForwardModel/TkSZForwardModel below)

#         r=None,
#         # explicit r values to evaluate the profile at. Required if Projection isn't given; ignored
#         # (and derived from Measurement+Projection instead) if it is.

#         measunit=None,
#         # conversion factor/Quantity from the projected physical profile to the observable's units.
#         # Leave as None to use the subclass's compute_measunit() (e.g. TSZForwardModel converts to
#         # compton-y, TkSZForwardModel converts to a uK temperature); pass a value here to override it.

#         verbose=False,
#         **cosmopars,
#         # fixed cosmological/astrophysical parameters shared across every submodel that needs them (H0, Om0, Ob0, sigma8, ns, XH, T_CMB, v_rms, etc.)
#         ):
#         self.verbose = verbose
#         self.cosmopars = cosmopars

#         self.Cosmology, self.TargetData, self.MapData, self.HaloModel = Cosmology, TargetData, MapData, HaloModel
#         self.Projection, self.Measurement, self.HOD, self.Spectra = Projection, Measurement, HOD, Spectra

#         self.configs = configs or {}

#         self.r = r
#         self.measunit = measunit

#     def _init(self, cls_or_obj, kwargs=None):
#         # Construct cls_or_obj(**kwargs) if it's an uninitialized class; otherwise return it as-is (already built)
#         return cls_or_obj(**(kwargs or {})) if isinstance(cls_or_obj, type) else cls_or_obj

#     def make_zdist(self, TargetData, TargetData_kwargs=None):
#         # TargetData: uninitialized class or an already-built targetdata object; make_zdist() is called on it either way
#         TargetData_kwargs = TargetData_kwargs or {}
#         targetdata = self._init(TargetData, TargetData_kwargs)
#         targetdata.make_zdist(**TargetData_kwargs)
#         zmean = targetdata.zMean if hasattr(targetdata, 'zMean') else targetdata.distave(targetdata.z, targetdata.dndz)
#         return targetdata, zmean

#     def make_Mhdist(self, targetdata, cosmology, TargetData_kwargs=None):
#         # needs cosmology for density unit calculations
#         targetdata.make_Mhdist(halomodel=cosmology, **(TargetData_kwargs or {}))
#         return targetdata

#     def make_rdist(self, cosmology, zmean, measurement=None, Projection=None, Projection_kwargs=None, r=None):
#         # Projection: uninitialized class (built from measurement+cosmology) or an already-built projection object.
#         # Skip Projection (and give r explicitly) to get just the mass/z-averaged 3D profile, with no projected observable
#         if Projection is None:
#             if r is None: raise ValueError("Give either a Projection class (with a Measurement) or an explicit r array")
#             return r, None

#         Projection_kwargs = Projection_kwargs or {}
#         if isinstance(Projection, type):
#             if measurement is None: raise ValueError("Measurement is required to build a Projection")
#             projection = Projection(Rs=measurement.R, AngDist=cosmology.dA(zmean), **Projection_kwargs)
#         else:
#             projection = Projection

#         r = np.geomspace(projection.r3D.min(), projection.r3D.max(), Projection_kwargs['nr'])
#         return r, projection

#     def make_projections(self, projection, r, mapdata):
#         proj2D = projection.proj2D(r)
#         beamconvolve = projection.beam_convolve(b_ell=mapdata.beam_ells, bTF=mapdata.beam_data, r_ell=mapdata.resp_ells, rTF=mapdata.resp_data)
#         CAP = projection.aperture_photometry(f_disc=np.sqrt(2))
#         return proj2D, beamconvolve, CAP

#     def make_halomodel(self, HaloModel, HaloModel_kwargs=None, **cosmopars):
#         # HaloModel: uninitialized class (e.g. Models.Codes.pyccl.HaloModel) or an already-built halomodel object
#         return self._init(HaloModel, (HaloModel_kwargs or {}) | cosmopars)

#     def make_distave(self, targetdata):
#         # averages a profile over the halo mass distribution, as an alternative to an HOD-weighted galaxy average
#         dndzdlogmhalo_norm = targetdata.dndlogMh*targetdata.dndz[:, None]/np.trapezoid(np.trapezoid(targetdata.dndlogMh, targetdata.logMh)*targetdata.dndz, targetdata.z)
#         return lambda prof: np.trapezoid(np.trapezoid(prof*dndzdlogmhalo_norm, targetdata.logMh), targetdata.z)

#     def make_HODave(self, HOD, Spectra, r, halomodel, targetdata, HOD_kwargs=None, Spectra_kwargs=None):
#         # HOD/Spectra: uninitialized classes or already-built objects
#         hod = self._init(HOD, HOD_kwargs)
#         spectra = Spectra(r, **(Spectra_kwargs or {})) if isinstance(Spectra, type) else Spectra
#         return hod, spectra, spectra.HODweighting(hod, halomodel, targetdata)

#     def make_profile(self, Profile, halomodel, r, z, logM, Profile_kwargs=None, **cosmopars):
#         # Profile: uninitialized class (e.g. Models.Papers.Popik2026.Pressure/Density) or an already-built profile object
#         if isinstance(Profile, type):
#             profile = Profile(rhoc=halomodel.rhoc, dndlogm=halomodel.dndlogm, bh=halomodel.bh, Plin=halomodel.Plin, **(cosmopars | (Profile_kwargs or {})))
#         else:
#             profile = Profile

#         prof = profile.total(r, z, logM)
#         prof1h = profile.onehalo(r, z, logM)
#         prof2h = profile.twohalo(r, z, logM)
#         return profile, prof, prof1h, prof2h

#     def make_project(self, proj2D, beamconvolve, CAP, measunit=1):
#         return lambda prof: measunit*CAP(*beamconvolve(proj2D(prof)))

#     def make_model(self, avemeth, prof, project=None):
#         # Full parameters -> mass/z-averaged 3D profile
#         aveprof = lambda params={}: avemeth(prof(params))
#         if project is None: return aveprof, aveprof
#         # Full parameters -> projected, beam-convolved, aperture-photometered observable
#         return aveprof, lambda params={}: project(avemeth(prof(params)))

#     def setup_profile(self):  # returns (profile, prof, prof1h, prof2h); implemented by subclasses
#         raise NotImplementedError("ForwardModel is profile-agnostic -- use TSZForwardModel, TkSZForwardModel, or another subclass that implements setup_profile()")

#     def compute_measunit(self):  # conversion factor from the projected physical profile to the observable's units; implemented by subclasses
#         raise NotImplementedError("ForwardModel is profile-agnostic -- use TSZForwardModel, TkSZForwardModel, or another subclass that implements compute_measunit(), or pass measunit explicitly")

#     def setup_forward_model(self):
#         # Runs the full pipeline using the submodels/configs stored on self by __init__
#         if self.verbose: print("Making Cosmology")
#         self.cosmology = self._init(self.Cosmology, self.cosmopars)

#         if self.verbose: print("Initializing Data")
#         self.mapdata = self._init(self.MapData, self.configs.get('MapData'))
#         self.measurement = None if self.Measurement is None else self._init(self.Measurement, self.configs.get('Measurement'))

#         if self.verbose: print("Making Redshift Distribution")
#         self.targetdata, self.zmean = self.make_zdist(self.TargetData, self.configs.get('TargetData'))

#         if self.verbose: print("Making Halo Mass Distribution")
#         self.targetdata = self.make_Mhdist(self.targetdata, self.cosmology, self.configs.get('TargetData'))

#         if self.verbose: print("Making Radius Distribution")
#         self.r, self.projection = self.make_rdist(self.cosmology, self.zmean, self.measurement, self.Projection, self.configs.get('Projection'), self.r)

#         if self.projection is not None:
#             if self.verbose: print("Setting up Projection")
#             self.proj2D, self.beamconvolve, self.CAP = self.make_projections(self.projection, self.r, self.mapdata)

#         if self.verbose: print("Setting up Halo Model")
#         self.halomodel = self.make_halomodel(self.HaloModel, self.configs.get('HaloModel'), **self.cosmopars)

#         if self.HOD is not None:
#             if self.verbose: print("Setting up HOD Average")
#             self.hod, self.spectra, self.avemeth = self.make_HODave(self.HOD, self.Spectra, self.r, self.halomodel, self.targetdata, self.configs.get('HOD'), self.configs.get('Spectra'))
#         else:
#             if self.verbose: print("Setting up Basic Average")
#             self.avemeth = self.make_distave(self.targetdata)

#         self.profile, self.prof, self.prof1h, self.prof2h = self.setup_profile()

#         self.measunit = self.measunit if self.measunit is not None else self.compute_measunit()

#         if self.verbose: print("Setting up Full Model")
#         project = None if self.projection is None else self.make_project(self.proj2D, self.beamconvolve, self.CAP, self.measunit)
#         self.aveprof, self.model = self.make_model(self.avemeth, self.prof, project)

#         return self


# class ComptonyForwardModel(ForwardModel):
#     def __init__(self, Pressure, **kwargs):
#         # Pressure: uninitialized Pressure profile class (e.g. Models.Papers.Popik2026.Pressure) or an already-built profile object
#         # its own config kwargs go under configs['Profile']
#         self.Pressure = Pressure
#         super().__init__(**kwargs)

#     def setup_profile(self):
#         if self.verbose: print("Setting up Pressure Profile Model", self.configs.get('Profile'))
#         return self.make_profile(self.Pressure, self.halomodel, self.r, self.targetdata.z, self.targetdata.logMh, self.configs.get('Profile'), **self.cosmopars)

#     def compute_measunit(self):
#         return pres_to_y(**self.cosmopars)
    
# class TtSZForwardModel(ForwardModel):
#     def __init__(self, Pressure, **kwargs):
#         # Pressure: uninitialized Pressure profile class (e.g. Models.Papers.Popik2026.Pressure) or an already-built profile object
#         # its own config kwargs go under configs['Profile']
#         self.Pressure = Pressure
#         super().__init__(**kwargs)

#     def setup_profile(self):
#         if self.verbose: print("Setting up Pressure Profile Model", self.configs.get('Profile'))
#         return self.make_profile(self.Pressure, self.halomodel, self.r, self.targetdata.z, self.targetdata.logMh, self.configs.get('Profile'), **self.cosmopars)

#     def compute_measunit(self):
#         return pres_to_y(**self.cosmopars)


# class TkSZForwardModel(ForwardModel):
#     def __init__(self, Density, **kwargs):
#         # Density: uninitialized Density profile class (e.g. Models.Papers.Popik2026.Density) or an already-built profile object
#         # its own config kwargs go under configs['Profile']
#         self.Density = Density
#         super().__init__(**kwargs)

#     def setup_profile(self):
#         if self.verbose: print("Setting up Density Profile Model", self.configs.get('Profile'))
#         return self.make_profile(self.Density, self.halomodel, self.r, self.targetdata.z, self.targetdata.logMh, self.configs.get('Profile'), **self.cosmopars)

#     def compute_measunit(self):
#         return rho_to_TkSZ(**self.cosmopars)



# class SZLikelihood(GaussianLikelihood, ForwardModel):
#     # Import the dictionary from the yaml file 
#     VERBOSE: bool = False
#     PAPERCHECK: bool = False
#     YAML_FILE: str | None = None
    
#     Measurement: Optional[Dict[str, Any]] = None  # Measurement Data
#     MapData: Optional[Dict[str, Any]] = None  # Map data (beams, responses, etc.)
#     TargetData: Optional[Dict[str, Any]] = None  # Target sample data (redshift and mass distribution)
    
#     HaloModel: Optional[Dict[str, Any]] = None  # Halo Model
#     HOD: Optional[Dict[str, Any]] = None  # Halo Occupancy Distribution
#     Projection: Optional[Dict[str, Any]] = None  # Projection method
#     Spectra: Optional[Dict[str, Any]] = None  # Spectra/Averaging Calculation
    
#     Pressure: Optional[Dict[str, Any]] = None  # Cluster Profile model
#     Density: Optional[Dict[str, Any]] = None  # Cluster Profile model
    
#     def format_yaml(self):
#         if self.VERBOSE: print("Loading in Fixed Parameters and Data/Model Dictionaries from yaml file")
#         # Load the yaml file
#         yaml_info = yaml_load_file(self.YAML_FILE)
#         # put shared dicts into each likelihood
#         for part in yaml_info['shared']: setattr(self, part, yaml_info['shared'][part])
#         # Get all fixed parameters in the params block that have a set value
#         self.cosmopars = {k: v["value"] for k, v in yaml_info['params'].items() if isinstance(v, dict) and "value" in v}

#     def import_modules(self):
#         for submodel in ['Measurement', 'Pressure', 'Density', 'Projection', 'TargetData', 'MapData', 'HOD', 'HaloModel', 'Spectra']:
#             if getattr(self, submodel) is None: continue
#             if self.VERBOSE: print("Importing and Initializing", submodel)
#             # Import the file of the submodel
#             if submodel=='HaloModel':
#                 module = importlib.reload(importlib.import_module(f"Models.Codes.{getattr(self, submodel)['name']}"))
#                 setattr(self, submodel.lower(), getattr(module, submodel))
#                 setattr(self, 'cosmology', getattr(module, 'Cosmology'))
#             else:
#                 module = importlib.reload(importlib.import_module(f"Models.Papers.{getattr(self, submodel)['name']}"))
#                 # Get the specific class and set it as an attribute
#                 setattr(self, submodel.lower(), getattr(module, submodel))
    
#     def initialize(self):
#         self.format_yaml()

#         self.import_modules()
                
#         # Initialize a cosmology, needed for some conversions in measurement and targetdata
#         self.cosmology = self.cosmology(**self.cosmopars)
 
#         # initialize Measurement, MapData, TargetData
#         self.measurement = self.measurement(**self.Measurement)
#         self.mapdata = self.mapdata(**self.MapData)
#         self.targetdata = self.targetdata(**self.TargetData)

#         self.setup_distributions()
#         self.setup_projections()
#         self.setup_halomodel()    
#         self.setup_HODave()
#         self.setup_profile()
        
#         self.setup_model()
        
#         self._get_data()

#     def get_requirements(self):
#         return {k: None for k in yaml_load_file(self.YAML_FILE)['params'].keys()}
        
#     def logp(self, **params_values):
#         theory = self._get_theory({**params_values})
#         return self.data.loglike(theory) 
    



# def pres_to_y(XH, **kwargs):  # factor to convert project Pressure to uK arcmin^2
#     return (c.sigma_T/c.m_e/c.c**2).cgs * (2+2*XH)/(3+5*XH) * u.g/u.Msun * u.Msun.to(u.g)

# def y_to_uK(nu, T_CMB, **kwargs):  # factor to convert compton y to uK arcmin^2
#     x = (c.h * nu / (c.k_B * T_CMB)).decompose().value
#     fnu = x / np.tanh(x / 2.0) - 4.0
#     return fnu*T_CMB*u.uK*1e6

# def pres_to_uK(nu, **kwargs):  # factor to convert project Pressure to uK arcmin^2
#     return pres_to_y(**kwargs)*y_to_uK(nu, **kwargs)

# class TSZLikelihood(SZLikelihood):   
#     def setup_profile(self):
#         if self.VERBOSE: print("Setting up Pressure Profile Model", self.Pressure)
#         self.pressure = self.pressure(rhoc=self.halomodel.rhoc, dndlogm=self.halomodel.dndlogm, bh=self.halomodel.bh, Plin=self.halomodel.Plin, **(self.cosmopars | self.Pressure))
        
#         self.prof = self.pressure.total(self.r, self.targetdata.z, self.targetdata.logMh)
#         self.prof1h = self.pressure.onehalo(self.r, self.targetdata.z, self.targetdata.logMh)
#         self.prof2h = self.pressure.twohalo(self.r, self.targetdata.z, self.targetdata.logMh)
            
#     def _get_theory(self, params_values):
#         return self.model(params_values).value
    
#     def _get_data(self):
#         if self.VERBOSE: print("Setting up Data")
#         # Get measurements
#         self.data = GaussianData("SZModel", self.measurement.R.value, self.measurement.y_data.value, self.measurement.y_cov.value)
#         # Units of measurement
#         self.measunit = pres_to_y(**self.cosmopars)
#         # elif self.meas.tSZ_data.unit==u.uK*u.arcmin**2:
#         #     self.measunit = self.halomodel.pres_to_uK(self.meas.freq, **self.cosmopars)
            
    
    
# def rho_to_TkSZ(v_rms, XH, T_CMB, **kwargs):  # factor to convert project density to uK arcmin^2
#     return v_rms * (c.sigma_T/c.m_p).cgs * (1+XH)/2 * T_CMB*u.uK*1e6 * u.cm.to(u.Mpc)**2 * u.Mpc**2/u.cm**2 * u.g/u.Msun * u.Msun.to(u.g)     
        
# class KSZLikelihood(SZLikelihood):       
#     def _get_theory(self, params_values):
#         return self.model(params_values).value
    
#     def _get_data(self):
#         if self.VERBOSE: print("Setting up Data")
#         # Get measurements
#         self.data = GaussianData("SZModel", self.measurement.R.value, self.measurement.TkSZ_data.value, self.measurement.TkSZ_cov.value)
#         # Units of measurement
#         self.measunit = rho_to_TkSZ(**self.cosmopars)

#     def setup_profile(self):
#         if self.VERBOSE: print("Setting up Density Profile Model", self.Density)
#         self.density = self.density(rhoc=self.halomodel.rhoc, dndlogm=self.halomodel.dndlogm, bh=self.halomodel.bh, Plin=self.halomodel.Plin, **(self.cosmopars | self.Density))
        
#         self.prof = self.density.total(self.r, self.targetdata.z, self.targetdata.logMh)
#         self.prof1h = self.density.onehalo(self.r, self.targetdata.z, self.targetdata.logMh)
#         self.prof2h = self.density.twohalo(self.r, self.targetdata.z, self.targetdata.logMh)








    
    # def test_timing(self):
    #     timings = {
    #         "profile1h": [],
    #         "profile2h": [],
    #         "profile_total": [],
    #         "average": [],
    #         "project": [],
    #         "forward_total": [],
    #         "cobaya": [],
    #         "loglike": [],
    #     }

    #     for i in range(10):
    #         time0 = time.time()

    #         prof = self.prof({})
    #         time1 = time.time()
    #         timings["profile_total"].append((time1 - time0) * 1000)

    #         profave = self.avemeth(prof)
    #         time2 = time.time()
    #         timings["average"].append((time2 - time1) * 1000)

    #         _ = self.project(profave)
    #         time3 = time.time()
    #         timings["project"].append((time3 - time2) * 1000)

    #         timings["forward_total"].append((time3 - time0) * 1000)

    #         theory = self._get_theory({})
    #         time4 = time.time()
    #         timings["cobaya"].append((time4 - time3) * 1000)

    #         _ = self.data.loglike(theory)
    #         time5 = time.time()
    #         timings["loglike"].append((time5 - time4) * 1000)
            
    #         prof2h = self.prof2h({})
    #         time6 = time.time()
    #         timings["profile2h"].append((time6 - time5) * 1000)
            
    #         prof1h = self.prof1h({})
    #         time7 = time.time()
    #         timings["profile1h"].append((time7 - time6) * 1000)

    #     results = {}
    #     for label, values in timings.items():
    #         mean_t = np.mean(values)
    #         median_t = np.median(values)
    #         results[label] = {"mean": mean_t, "median": median_t}
    #         print(f"{label}: mean={mean_t:.2f} ms, median={median_t:.2f} ms")

    #     return results