"""
Effects of Galaxy Formation on Thermodynamics of the Intracluster Medium

arxiv.org/pdf/astro-ph/0703661
ui.adsabs.harvard.edu/abs/2007ApJ...668....1N
"""


from config import *
thispath = os.path.dirname(os.path.abspath(__file__))


from Models.Papers.PlotsTables import BasePlots2, ParamTable
from Models.HaloModels import pyccl_model


class Cosmology(pyccl_model):  
    definedparams = {
    # 3. We adopt fb = 0.175 and h = 0.7 throughout this work. Note thatΩM = 0.3 and ΩΛ = 0.7 are assumed in both analyses.
    "Fb": 0.175, "h": 0.7, "Om0": 0.3, "Ol0": 0.7,
    # 4. Note that we use μ = 0.59 and μe = 1.14 throughout this work.
    "mu": 0.59, "mu_e": 1.14,
    # 2. In this study, we analyze high-resolution cosmological simulations of 16 cluster-sized systems in the flat ΛCDM model:
    "Om0": 0.3, "Ob0": 0.04286, "sigma8": 0.9,
    }
    def __init__(self, inputdict={}, **inputvars):
        super().__init__(self.definedparams | inputdict | inputvars)
        for key, value in (self.definedparams | inputdict | inputvars).items(): setattr(self, key, value)


class HaloModel():
    # 2. The masses are reported at the radius r500 enclosing overdensities with respect to the critical density at the redshift of the output.
    MassDef = '500c'
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
    
    def R500c(self, **kwargs):
        return ((10**self.logM500c*u.Msun/(4/3*np.pi*self.rho_crit(self.z)))**(1/3)).to(u.Mpc)

    
class ClusterProfile():
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        try: self.p0 = TableA1().getparams(Runs=self.Runs,Sample=self.Sample).to_dict()
        except: self.p0 = {}
        
    # 4. The temperature, entropy (defined as K ≡ kBT/n2/3e ), and pressure profiles are normalized to the values computed for the given cluster mass using a simple self-similar model (Kaiser 1986; Voit 2005):
    def P500(self,  **kwargs):  # Eq 3
        return 1.45e-11*u.erg/u.cm**3 * (10**self.logM500c/1e15)**(2/3) * (self.H(self.z)/self.H0)**(8/3)

    # A. we parameterize the pressure profile using the generalized NFW model where x ≡ r/rs, rs = r500/c500, P500 is given by equation 3, and (α, β, γ) are the slopes at r ∼ rs, r ≫ rs, and r ≪ rs, respectively.
    def Pr_over_P500(self, pdict={}, **kwargs):  # Eq A1
        p = self.p0 | pdict | kwargs
        
        rs = self.R500c/p['c500']
        x = self.r/rs
        
        return p['P0'] / (x**p['gamma'] * (1+x**p['alpha'])**((p['beta']-p['gamma'])/p['alpha']))
    
    
# TABLE A1 Best-fit parameters of the pressure profile.
class TableA1(ParamTable):
    def __init__(self, filename=f"{thispath}/TableA1.csv"):
        super().__init__(filename)
        

# Fig. 1.— Radial profiles of the ICM in relaxed simulated clusters at z = 0. For each of the physical profiles the upper panels show the profiles, while the bottom panels show the corresponding fractional deviations of the profiles in the CSF simulation from the corresponding profiles in the non-radiative runs. The figure shows gas density (top-left), temperature (top-right), entropy (bottom-left), and pressure (bottom-right) profiles. Thick solid and dashed lines show the average profiles of the relaxed clusters in the CSF and non-radiative runs, respectively. The shaded band indicates the rms scatter around the mean profile for the CSF run. In addition, the dashed and dotted lines indicate the average profiles of systems with TX > 2.5 and < 2.5 keV, respectively, in the CSF simulations. Note that the entropy profiles of the non-radiative runs outside 0.3r500 are well-described by a power-law radial profile K ∝ r1.2, indicated by the dashed line in the bottom-left panel.
class Fig1d(BasePlots2):
    subplots = [[
        dict(name='Fig1d', filename='Fig1d', figsize=(5.5, 5),
             xlabel=r'$r/r_{500}$', xlim=(4.5e-2, 2.3), xscale='log',
             ylabel=r'$P(r)/P_{500}$', ylim=(2.4e-3, 3.8e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Fig. 2.— Comparison of the ICM profiles in relaxed clusters at the present day (z ≈ 0) in cosmological cluster simulations and the Chandra sample of Vikhlinin et al. (2006). The panels show the gas density (top-left), temperature (top-right), entropy (bottom-left), and pressure (bottom-right). Thick solid and dashed lines show the mean profiles in the CSF and non-radiative simulations, respectively, while the observed profiles are shown by the thin dotted, long-dashed and short-dashed lines for the systems with TX > 5 keV, 2.5 < TX < 5 keV, and TX < 2.5 keV, respectively. Note that at r & 0.1r500 the profiles of the CSF simulations provide a better match to the observed profiles than the profiles in the non-radiative runs.
class Fig2d(BasePlots2):
    subplots = [[
        dict(name='Fig2d', filename='Fig2d', figsize=(5.5, 5),
             xlabel=r'$r/r_{500}$', xlim=(4.5e-2, 2.3), xscale='log',
             ylabel=r'$P(r)/P_{500}$', ylim=(2.4e-3, 1.9e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# Fig. A1.— Generalized NFW fits to the pressure profiles of relaxed clusters in simulations and Chandra X-ray observations. Open circles and squares show the mean profiles in the CSF and non-radiative simulations, respectively. Thin dotted shows Chandra X-ray clusters with TX > 5 keV. Thick lines show the best-fit generalized NFW model to simulation and observed profiles (see Table A1 for the best-fit parameters).
class FigA1(BasePlots2):
    subplots = [[
        dict(name='FigA1', filename='FigA1', figsize=(5.5, 5),
             xlabel=r'$r/r_{500}$', xlim=(4.5e-2, 2.3), xscale='log',
             ylabel=r'$P(r)/P_{500}$', ylim=(2.4e-3, 1.9e2), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)



class ParamsTable(ParamTable):  # Table A1
    def __init__(self, filename=f"{thispath}/params.csv"):
        super().__init__(filename)

