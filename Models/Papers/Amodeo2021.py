"""
Atacama Cosmology Telescope: Modeling the gas thermodynamics in BOSS CMASS galaxies from kinematic and thermal Sunyaev-Zel'dovich measurements

ui.adsabs.harvard.edu/abs/2021PhRvD.103f3514A
arxiv.org/pdf/2009.05558
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))




class Cosmology():
    # IIA. τgal refers to the optical depth of the galaxy group considered, and vr = 1.06 × 10−3c is the RMS of the peculiar velocities, projected along the line of sight, where the magnitude adopted is for the median redshift of the
    v_rms = 1.06e-3
    
    # I. We adopt a flat ΛCDM cosmology with matter densityΩm = 0.25, baryon density Ωb = 0.044, dark energy density ΩΛ = 0.75, and local expansion rate H0 = 70km s−1 Mpc−1 (h ≡ H0/(100 km s−1 Mpc−1).
    p0 = {'Om0': 0.25, 'Ob0': 0.044, 'OL0': 0.75, 'H0': 70, 'h': 0.7}

    # from mopc-g-gt, mopc.py
    T_CMB = 2.725*u.K # CMB temperature from mopc-g-gt, mopc.py
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)


    # I. critical density of the universe at the halo’s redshift
    def rhocrit(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return (3*self.H0**2 * (self.Om0*(1+self.z)**3+self.OL0) / (8*np.pi*c.G)).to(u.Msun/u.Mpc**3)

    # baryon fraction fb = Ωb/Ωm
    def fb(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return p['Ob0']/p['Om0']


class HaloModel(Cosmology):
    # App A. dn/dM is the mass function of the neighboring halos that we compute from [Sheth 2002]
    MassFunc = 'sheth99'
    # App A. b(M ) is the linear bias factor of the halo of mass M , from [Sheth 2001]
    HaloBias = 'sheth01'

    # App A. Halo masses are quoted as M200, at a radius of R200, within which the halo density is 200 times the critical density of the universe at the halo’s redshift
    MassDef = '200c'
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
    # R200 for a halo of mass M200 (M200 = (4/3) pi 200 rho_cr(z) R200^3)
    def R200(self, pdict={}, **kwargs):
        return ((10**self.logM200*u.Msun)/(4/3*np.pi*200*self.rhocrit(pdict, **kwargs)))**(1/3)


class HaloProfiles(HaloModel):
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

        try: self.p0 = Cosmology.p0 | GNFWParams().df.iloc[0].to_dict()
        except: self.p0 = Cosmology.p0

    # core scale x_c,t for the pressure profile, fixed following [54]
    def xct(self, pdict={}, **kwargs):
        return 0.497 * (10**self.logM200/1e14)**(-0.00865) * (1+self.z)**0.731

    # Eq 16. Generalized NFW profile (dimensionless shape) for the gas density, x = r/R200.
    # gamma_k = -0.2 and alpha_k = 1 are fixed, following [76].
    def rho_GNFW(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        x = self.r/self.R200(pdict, **kwargs)
        gammak, alphak = -0.2, 1
        return 10**p['log10rho0'] * (x/p['xck'])**gammak * (1+(x/p['xck'])**alphak)**(-(p['betak']-gammak)/alphak)

    # Eq 16. rho_gas(x) = rho_GNFW(x) * rho_cr(z) * fb
    def rho_gas(self, pdict={}, **kwargs):
        return (self.rho_GNFW(pdict, **kwargs) * self.rhocrit(pdict, **kwargs) * self.fb(pdict, **kwargs)).to(u.g/u.cm**3)

    # Eq 17. Modified GNFW profile (dimensionless shape) for the thermal pressure, x = r/R200.
    # gamma_t = -0.3 is fixed, following [54, 75].
    def P_GNFW(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        x = self.r/self.R200(pdict, **kwargs)
        gammat = -0.3
        xct = self.xct(pdict, **kwargs)
        return p['P0'] * (x/xct)**gammat * (1+(x/xct)**p['alphat'])**(-p['betat'])

    # P200 = G M200 200 rho_cr(z) fb / (2 R200)
    def P200(self, pdict={}, **kwargs):
        return (c.G * 10**self.logM200*u.Msun * 200*self.rhocrit(pdict, **kwargs)*self.fb(pdict, **kwargs) / (2*self.R200(pdict, **kwargs))).to(u.erg/u.cm**3)

    # Eq 17. P_th(x) = P_GNFW(x) * P200
    def P_th(self, pdict={}, **kwargs):
        return self.P_GNFW(pdict, **kwargs) * self.P200(pdict, **kwargs)


# Appendix B: Dust emission
class Dust_new(Cosmology):
    z0 = 0.55 # redshift of the dust emitters, II.A.p1
    wavelength0 = 350*u.um
    wavelength0 = (c.c/wavelength0).to(u.GHz) # rest-frame frequency at which we normalize the dust emission, assumed from matching Fig11 I(v) plots
    param_units = {'A_dust':u.kJy/u.sr, 'T_dust': u.K,'c_1': 1/u.arcmin, 'c_2': 1/u.arcmin**2}
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

        try: self.p0 = DustParams().getparams(Data=self.Data).to_dict()
        except: self.p0 = {}
        self.p0 = self.p0 | {'z0': self.z0, 'nu0': self.nu0}
        
        try: self.nu = (c.c/(self.wavelength)).to(u.GHz)
        except: print("no")
    
    # We model boththe frequency and spatial distribution of the dust emission as Eq B1 where ν0 is the rest-frame frequency at which we normalize the dust emission, R is the radius of the aperture photometry filter, z is the redshift of the dust emitters,Adust is the amplitude of the dust emission in [kJy/sr],βdust is the dust spectral index, Tdust is the dust temperature in K, and c0, c1, c2 are the polynomial coefficients parametrizing the radial profile
    def x(self, nu, T, **kwargs):
        return ((nu)*c.h/c.k_B/(T)).decompose().value
    
    def dustpoly(self, pdict={}, **kwargs): # Eq B1, Polynominal dust fit to stacked profiles I(nu) [in jr/sr]
        p = {p: k*self.param_units[p] if p in self.param_units else k for p, k in (self.p0 | pdict | kwargs).items()}
        # Amplitude part
        amp = p['A_dust']*(self.nu*(1+p['z0'])/(p['nu0']))**(p['beta_dust']+3)
        # Planck function part
        planck = (np.exp(self.x(self.nu*(1+p['z0']), p['T_dust']))-1)/(np.exp(self.x(self.nu*(1+p['z0']), p['T_dust']))-1)
        # Polynomial part
        poly = p['c_0']+p['c_1']*self.R+p['c_2']*self.R**2
        # Combine all parts
        return amp*planck*poly

    # Conversion of polynomial fit to uK arcmin^2
    def dust_uKarcmin(self, R, nu, **kwargs):
        dB_dT = ((2*c.h*(nu)**3/c.c**2)).to(u.kJy) * x/self.T_CMB * np.exp(x)/(np.exp(x)-1)**2  # Planck function for unit conversion to K
        dustprof = lambda p: self.dustpoly(R, nu)(p)/dB_dT*1e6 * np.pi*R**2  # Also multipy by area of disc
        return lambda p={}: dustprof(self.p0 | p).value
    
    # Conversion of polynomial fit to uK arcmin^2, TODO 2
    def dust_y(self, R, nu, **kwargs):
        x = (c.h * nu / (c.k_B * self.T_CMB*u.K)).decompose().value
        fnu = x / np.tanh(x / 2.0) - 4.0
        return lambda p={}: self.dust_uKarcmin(R, nu)(self.p0 | p) / (fnu*self.T_CMB*1e6)
    
    
# Best-fit dust model parameters (Eq. B1), fit to Herschel-only (Hr) and ACT+Herschel (ACTHr) data, estimated from Fig. 11a.
class DustParams(ParamTable):
    def __init__(self, filename=f"{thispath}/DustParams.csv"):
        super().__init__(filename)


# TABLE II. Best-fit values of the GNFW two-halo term parameters (Eq. 7) for the gas density and pressure profiles beyond ~2R200c, obtained from the joint fit to the kSZ and tSZ measurements.
class GNFWParams(ParamTable):
    def __init__(self, filename=f"{thispath}/Table2.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


# FIG. 6. Top: comparison of our best-fit gas density (left) and thermal pressure (right) profiles (blue curves and 2σ bands) with the related profiles from two cosmological simulations: [46] (magenta) and Illustris/TNG [28] (orange and green), and a NFW profile [20] (black). For the pressure profile, we also show the Planck 2013 [87] best fit (maroon dotted line). We show average profiles, where each halo contribution is weighted by its mass according to the mass probability density function of the CMASS catalog used in this work, and at the same redshift (z = 0.55). We select red galaxies from TNG and show both stellar mass(orange) and halo mass-weighted average profiles. The vertical grey lines enclose the range where we measure the kSZ and the tSZ. Middle: projected density and pressure profiles, for comparison purposes. Bottom: comparison of the profiles projected into the kSZ (left) and tSZ (right) observable space with the measurements by [18] in the ACT f150 band (blue points and 1σerror bars). The projections of the simulated and the NFW profiles account for the convolution with the ACT beam and the aperture photometry filtering, as described in Section II. The black dashed curve shows the NFW profile truncated at the virial radius. The tSZ simulated profiles also include the dust correction from our ACT+Herschel measurements (2σ).
class Fig6(BasePlots2):
    subplots = [
        [dict(name='Fig6a', filename='Fig6a', figsize=(8, 6),
              xlabel=r'$R [\text{Mpc}]$', xlim=(7.7e-2, 1.25e1), xscale='log',
              ylabel=r'$\rho_\text{gas} \ [\text{g cm}^{-3}]$', ylim=(5e-32, 4.1e-26), yscale='log'),
         dict(name='Fig6b', filename='Fig6b', figsize=(8, 6),
              xlabel=r'$R [\text{Mpc}]$', xlim=(7.7e-2, 1.25e1), xscale='log',
              ylabel=r'$P_\text{th} \ [\text{erg cm}^{-3}]$', ylim=(1.5e-16, 2.5e-12), yscale='log')],
        [dict(name='Fig6c', filename='Fig6c', figsize=(8, 6),
              xlabel=r'$R [\text{arcmin}]$', xlim=(0.8, 6.1), xscale='linear',
              ylabel=r'$\rho^\text{2D}_\text{gas} \ [\text{g cm}^{-3}] \cdot \text{Mpc}$', ylim=(6.9e-5, 1.5e-3), yscale='log'),
         dict(name='Fig6d', filename='Fig6d', figsize=(8, 6),
              xlabel=r'$R [\text{arcmin}]$', xlim=(0.8, 6.1), xscale='linear',
              ylabel=r'$P^\text{2D}_\text{th} \ [\text{erg cm}^{-3}] \cdot \text{Mpc} $', ylim=(3e10, 9.5e11), yscale='log')],
        [dict(name='Fig6e', filename='Fig6e', figsize=(7.5, 5),
              xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
              ylabel=r'$T_\text{kSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', ylim=(1.2e-1, 4e1), yscale='log'),
         dict(name='Fig6f', filename='Fig6f', figsize=(7.5, 5),
              xlabel=r'$R [\text{arcmin}]$', xlim=(0.75, 6.25), xscale='linear',
              ylabel=r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', ylim=(-22, 1.25), yscale='linear')],
    ]

    def __init__(self):
        super().__init__(thispath)


# FIG. 7. One and two halo terms contributing to the density (left) and pressure (right) profiles of a halo of 3 × 1013M atz = 0.55. The contribution of the two-halo term is not negligible above ∼ 2R200c.
class Fig7(BasePlots2):
    subplots = [[
        dict(name='density', filename='Fig7a', figsize=(4, 3),
             xlabel=r'$r/R_\text{200c}$', xlim=(7.3e-3,3.2e1), xscale='log',
             ylabel=r'$\rho_\text{gas}(r) / \rho_c$', ylim=(2.7e-4,8.2e3), yscale='log'),
        dict(name='pressure', filename='Fig7b', figsize=(4, 3),
             xlabel=r'$r/R_\text{200c}$', xlim=(7.3e-3,3.2e1), yscale='log',
             ylabel=r'$P_\text{th} (r) / P_{200}$', ylim=(5.5e-8,8.2e1), xscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 11. Top: Constraints on the dust model (B1). Results obtained with the simultaneous fit to the GNFW pressure model using ACT and Herschel temperature measurements are shown in blue. Results from the fit of the dust model to Herschel data only are shown in orange. All parameters match within 1σ. Middle: profiles measured in the three Herschel/SPIRE frequency bands (black) in intensity units [kJy/sr]. The blue and orange curves show the best-fit models to Herschel+ACT and Hershel only data, respectively, and the corresponding bands show the 2σ (2nd-98th percentiles) of the distribution of the models obtained from the MCMC chains. Bottom: profiles measured in the two ACT frequency bands (black) in cumulative temperature units of [μK · arcmin2]. The blue curves and 2σ bands show results obtained with the simultaneous fit of the dust and the GNFW pressure models using Herschel+ACT data. We separate the dust contribution (dot-dashed curves) from the thermal pressure (dashed curves). The best-fit model is the sum of the two contributions.
# Note: row 0 (middle panels, 3 cols) and row 1 (bottom panels, 2 cols) are not
# a rectangular grid, so use .row(0)/.row(1) here rather than .full()/.col().
class Fig11(BasePlots2):
    subplots = [
        [dict(name='Fig11b', filename='Fig11b', figsize=(6.7, 5),
              xlabel=r'$R [\text{arcmin}]$', xlim=(1.5, 6.2), xscale='linear',
              ylabel=r'$I \ [\text{kJy/sr}]$', ylim=(-0.2, 1.6), yscale='linear'),
         dict(name='Fig11c', filename='Fig11c', figsize=(6.7, 5),
              xlabel=r'$R [\text{arcmin}]$', xlim=(1.5, 6.2), xscale='linear',
              ylabel=r'$I \ [\text{kJy/sr}]$', ylim=(-0.3, 2.8), yscale='linear'),
         dict(name='Fig11d', filename='Fig11d', figsize=(6.7, 5),
              xlabel=r'$R [\text{arcmin}]$', xlim=(1.5, 6.2), xscale='linear',
              ylabel=r'$I \ [\text{kJy/sr}]$', ylim=(-0.3, 2.7), yscale='linear')],
        [dict(name='Fig11e', filename='Fig11e', figsize=(7.5, 5),
              xlabel=r'$R [\text{arcmin}]$', xlim=(1.5, 6.2), xscale='linear',
              ylabel=r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', ylim=(-22.5, 3.5), yscale='linear'),
         dict(name='Fig11f', filename='Fig11f', figsize=(7.5, 5),
              xlabel=r'$R [\text{arcmin}]$', xlim=(1.5, 6.2), xscale='linear',
              ylabel=r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', ylim=(-27.5, 2.5), yscale='linear')],
    ]

    def __init__(self):
        super().__init__(thispath)
        
        
        
        
"""Everything below this line is old implementation which I'll be updating"""

from Models.Studies import BaseStudy
class Study(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021PhRvD.103f3514A
    subs = {}
    info = {
        # cosmological parameters, Ip10, IIA.Ap4/p5
        'Om0': 0.25, 'Ob0': 0.044, 'OL0': 0.75, 'h': 0.7,  
        'v_rms': 1.06e-3, 'XH':0.76,   # RMS of peculiar velocites [v/c] and hydrogen mass fraction
        'T_CMB': 2.725*u.K,  # mop-c-gt, mopc.py
        # Model implementation, Ip10, Appendix A p2
        'MassDef': 'fof', 'MassFunc':'sheth99', 'HaloBias':'sheth01',
    }

from Models.Dust import BaseDust
class Dust(BaseDust, Study):  # arxiv.org/abs/2009.05558
    models = {'fitdata': ['Hr', 'ACTHr'],}  # Fit to Hershel or Act and Hershel
    params = {
        # Best-fit dust params, all estimated from Fig11a, # TODO: get more accurate
        'A_dust': {'Hr': 0.326, 'ACTHr': 0.363},  # amplitude of dust emission [kJy/sr]
        'T_dust': {'Hr': 20.7,  'ACTHr': 16.9},   # Dust temperature [K]
        'beta_dust':{'Hr': 1.13, 'ACTHr': 1.13},  # Dust spectral index
        'c_0': {'Hr': 5.00,'ACTHr': 6.046},  # Polynomial coefficient on x^0
        'c_1': {'Hr': -1.48, 'ACTHr': -1.88}, # Polynomial coefficient on x^1
        'c_2': {'Hr': 0.113, 'ACTHr': 0.148}, # Polynomial coefficient on x^2
        # Fixed dust params
        'z0': 0.55,  # redshift of the dust emitters, II.A.p1
        'nu0': ((c.c/(350*u.um)).to(u.GHz)).value,  # rest-frame frequency at which we normalize the dust emission, assumed from matching Fig11 I(v) plots
    }
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        self.require(['fitdata'])
    
    # Polynominal dust fit to stacked profiles I(nu) [in jr/sr]
    def dustpoly(self, R, nu, **kwargs):
        x = lambda nu, p: ((nu)*c.h/c.k_B/(p['T_dust']*u.K)).decompose()
        planck = lambda p: (np.exp(x(self.p0['nu0']*u.GHz, p))-1)/(np.exp(x(nu*(1+self.p0['z0']), p))-1)  # Planck function part
        amp = lambda p: p['A_dust']*u.kJy/u.sr*(nu*(1+self.p0['z0'])/(self.p0['nu0']*u.GHz))**(p['beta_dust']+3)  # Amplitude part
        poly = lambda p: p['c_0']+p['c_1']*R+p['c_2']*R**2  # Polynomial part
        dustfunc = lambda p: amp(p)*planck(p)*poly(p)  # Combine
        return lambda p={}: dustfunc(self.p0 | p)

    # Conversion of polynomial fit to uK arcmin^2
    def dust_uKarcmin(self, R, nu, **kwargs):
        x = ((nu)*c.h/c.k_B/(self.T_CMB*u.K)).decompose().value
        dB_dT = ((2*c.h*(nu)**3/c.c**2)).to(u.kJy) * x/self.T_CMB * np.exp(x)/(np.exp(x)-1)**2  # Planck function for unit conversion to K
        dustprof = lambda p: self.dustpoly(R, nu)(p)/dB_dT*1e6 * np.pi*R**2  # Also multipy by area of disc
        return lambda p={}: dustprof(self.p0 | p).value
    
    # Conversion of polynomial fit to uK arcmin^2, TODO 2
    def dust_y(self, R, nu, **kwargs):
        x = (c.h * nu / (c.k_B * self.T_CMB*u.K)).decompose().value
        fnu = x / np.tanh(x / 2.0) - 4.0
        return lambda p={}: self.dust_uKarcmin(R, nu)(self.p0 | p) / (fnu*self.T_CMB*1e6)