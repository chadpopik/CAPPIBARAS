"""
On the Cluster Physics of Sunyaev-Zel'dovich and X-Ray Surveys. II. Deconstructing the Thermal SZ Power Spectrum

ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
arxiv.org/pdf/1109.3711
"""


from config import *
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))


class Cosmology():
    # 2. We adopt a flat tilted ΛCDM cosmology, with total matter density (in units of the critical) Ωm = ΩDM + Ωb = 0.25, baryon density Ωb = 0.043, cosmological constant ΩΛ = 0.75, a present day Hubble constant of H0 = 100h km s−1 Mpc−1, a scalar spectral index of the primordial power-spectrum ns= 0.96 and σ8 = 0.8.
    Om0 = 0.25
    Ob0 = 0.043
    Ol0 = 0.75
    H0 = 100
    ns = 0.96
    sigma8 = 0.8
    
    # 2. It is important to note that all masses and distances quoted in this work are given relative to
    h = 0.7
    
    # 3. where XH = 0.76 is the primordial hydrogen mass fraction
    XH = 0.76


class HaloModel():
    # 2. We adopt the standard working definition of cluster radii R∆as the radius at which the mean interior density equals ∆ times the critical density, ρcr(z) (e.g., for ∆ = 200 or 500).
    mdef = '200c'


# The normalized average pressure profiles and parametrized fits to these profiles from simulations with AGN feedback scaled by (r/R200)3, in mass bins (left panel) and redshift bins (right panel). Here we have independently fit each mass and redshift bin.
class Fig1(BasePlots2):
    subplots = [[
        dict(name='Fig1a', filename='Fig1a', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
        dict(name='Fig1b', filename='Fig1b', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2a', filename='Fig2a', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
        dict(name='Fig2b', filename='Fig2b', figsize=(8, 6),
             xlabel=r'$r/R_{200}$', xlim=(3e-2, 3), xscale='log',
             ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$', ylim=(8e-4, 2.35e-1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)
