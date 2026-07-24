"""
Atacama Cosmology Telescope: Combined kinematic and thermal Sunyaev-Zel'dovich measurements from BOSS CMASS and LOWZ halos

ui.adsabs.harvard.edu/abs/2021PhRvD.103f3513S
arxiv.org/pdf/2009.05557
"""


from config import *
thispath = os.path.dirname(os.path.abspath(__file__))


from Models.Papers.PlotsTables import BasePlots2, ParamTable, read_wide_table
from Models.HaloModels import pyccl_model


class Cosmology():
    # 1. We convert the kSZ temperatures into integrated optical depth to Thomson scattering in the CAP filter viaTkSZ = τCAPTCMB(vtrue rms /c), with TCMB = 2.726K and vtrue rms = 313 km/s at z = 0.55, according to linear theory.
    defined ={
        'T_CMB':2.726,  # CMB temp [K], Section F.1p8
        'v_rms': {'lowz':320, 'cmass':313},  # rms velocity [km/s] at mean redshifts, Section F.1p8/F.2p1
    }
    pass

class HaloModel():
    pass


# FIG. 2. Redshift distribution of the LOWZ K (DR10), CMASS K (DR10) and CMASS M (DR12) spectroscopic galaxies whose positions on the sky overlap with the ACT DR5 microwave maps. The mean redshifts are 0.31 for LOWZ K and 0.54 for CMASS K and CMASS M. They are indicated by the vertical dashed lines.
class Fig2(BasePlots2):
    subplots = [[
        dict(name='Fig2', filename='Fig2', figsize=(6, 4),
             xlabel=r'$z$', xlim=(0, 0.7), xscale='linear',
             ylabel=r'$N_\text{galaxies}$', ylim=(0, 3.2e4), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 3. Host halo virial masses of the LOWZ K (DR10), CMASS K (DR10) and CMASS M (DR12) galaxies, as inferred from their stellar masses in Appendix G. The dashed lines indicate the mean halo masses for each sample, 〈Mvir〉 = 3 × 1013M for CMASS K and 〈Mvir〉 = 5 × 1013M for LOWZ K. These do not coincide with the modes of the mass distributions, due to the high mass tails (the x-axis is logarithmic). In this analysis, we further discard the objects withMvir > 1014M to avoid tSZ contamination to the kSZ signal, as explained in Sec. IV E.
class Fig3(BasePlots2):
    subplots = [[
        dict(name='Fig3', filename='Fig3', figsize=(5, 4),
             xlabel=r'$M_\text{vir} \ [M_\odot]$', xlim=(1e11, 1e15), xscale='log',
             ylabel=r'$N_\text{galaxies}/N_\text{total}$', ylim=(0, 6.9e-14), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 5. The effective beam profiles for the coadded f90 and f150 DR5 maps from [76] are shown in solid blue and red, and compared to Gaussian beams with the same FWHM. Percentlevel sidelobes are visible at 2–4′. These are included in the modeling of the signal in [36]. The beams for the ILC maps with and without deprojection from [78] are shown in green and cyan. These are Gaussian by construction.
class Fig5(BasePlots2):
    subplots = [[
        dict(name='Fig5', filename='Fig5', figsize=(6, 4),
             xlabel=r'$\theta \ [\text{arcmin}]$', xlim=(8e-2, 1.25e1), xscale='log',
             ylabel=r'$B(\theta)/B(0)$', ylim=(1e-3, 2e0), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 7. Top: The mean CMASS kSZ signal in each compensated aperture photometry filter with radius R (see Eq. (11)), obtained by stacking the single-frequency temperature maps f90 and f150. The joint best fit kSZ profile from [36], convolved with the beams of f90 and f150, is shown in solid lines. The kSZ signal is detected at 7.9 σ(i.e. SNRmodel = √∆χ2 = 7.9). The dashed lines show the expected kSZ signal if the gas followed the dark matter (NFW) profile (convolved with the beams and CAP filters). The data show that the electron profile is more extended than the dark matter profile at very high significance (√χ2 NFW − χ2 best fit = 96). The vertical lines show the halo virial radius (1.6′ at z = 0.55) added in quadrature with the beam standard deviations (σ = FWHM/√8 ln 2 = 0.55′ in f150 and 0.89’ in f90). To guide the eye, the gray solid lines correspond to Gaussian profiles with FWHM = 1.3′ (f150 beam), FWHM = 2.1′ (f90 beam) and FWHM = 6′ (similar to the measured profile) from left to right. They are normalized to match the largest aperture in f150. The y-axis on the right converts the measured kSZ signal into the CAP optical depth to Thomson scattering, which counts the number of free electrons within the CAP filter. Null tests are shown in Figs. 20 and 21. Bottom panel: correlation matrix between the different CAP filters and frequencies.
class Fig7a(BasePlots2):
    subplots = [[
        dict(name='Fig7a', filename='Fig7a', figsize=(5, 4),
             xlabel=r'$R \ [\text{arcmin}]$', xlim=(0.75, 6.3), xscale='linear',
             ylabel=r'$T_\text{kSZ} [\mu \text{K} \cdot \text{arcmin}^2]$', ylim=(3e-2, 7e1), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 9. Mean tSZ + dust signal in all compensated aperture photometry filters, as defined in Equation 10. These were obtained by stacking on the single-frequency temperature maps f90 and f150. The best joint fit tSZ+dust profile to the f90, f150 and Herschel data from [36] is shown at these frequencies in solid lines. The no-signal hypothesis is rejected at 18.9 σ (see Table I). The impact of dust emission is seen in the difference between these profiles and Fig. 8, not at the large apertures where the noise is different, but at the smallest apertures where the dust signal fills in the tSZ decrement (causing even a “negative tSZ decrement” at 150 GHz). The vertical lines show the halo virial radius (1.6′ atz = 0.55) added in quadrature with the beam standard deviations (σ = FWHM/√8 ln 2 = 0.55′ in f150 and 0.89’ in f90). The correlation matrix for the different CAP filters and frequencies is identical to Fig. 7.
class Fig9(BasePlots2):
    subplots = [[
        dict(name='Fig9', filename='Fig9', figsize=(5, 4),
             xlabel=r'$R \ [\text{arcmin}]$', xlim=(0.75, 6.3), xscale='linear',
             ylabel=r'$T_\text{tSZ+dust} [\mu \text{K} \cdot \text{arcmin}^2]$', ylim=(-25, 0.3), yscale='log'),
    ]]

    def __init__(self):
        super().__init__(thispath)


# FIG. 31. Stellar mass estimates of the LOWZ K (DR10), CMASS K (DR10) and CMASS M (DR12) galaxies from [66] for CMASS and from the Wisconsin group. The dashed lines indicate the mean masses for each sample.
class Fig31(BasePlots2):
    subplots = [[
        dict(name='Fig31', filename='Fig31', figsize=(5, 4),
             xlabel=r'$M_* \ [M_\odot]$', xlim=(2e10, 2e12), xscale='log',
             ylabel=r'$N_\text{galaxies}/N_\text{total}$', ylim=(0, 5.45e-12), yscale='linear'),
    ]]

    def __init__(self):
        super().__init__(thispath)


class StudiesInfoTable(ParamTable):  # rms velocities, halo masses, redshifts, galaxy counts, per sample
    def __init__(self, filename=f"{thispath}/studies_info.csv"):
        self.df = read_wide_table(filename)

