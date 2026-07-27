"""
Searching for Systematics in Forward Modeling Sunyaev-Zeldovich Profiles

ui.adsabs.harvard.edu/abs/2023arXiv230710919M
arxiv.org/pdf/2307.10919
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Moser2023")




"""Old implementation being phased out"""

from Models.Projections import BaseProjection
class Moser2023(BaseProjection):  # https://arxiv.org/abs/2307.10919
    info = {'r_min': 1e-3, 'r_max':10,  # line of sight range, tested in 2.1
            'NNR': 100,  # fineness of R values for summing signal in aperture photometry
            'resolution_factor': 3.5,  # additional factor of fineness for R values in RFT
            'lmin': 170.0, 'lmax': 1.4e6,  # ell range going into the FHT, 2.3.2
            'pad': 100,  # number of points padded on sides of range to help accuracy
            'sizeArcmin': 30.0,  # size of FFT map, 2.3.1
            'disc_fac': np.sqrt(2),  # standard for aperture photometry
            }
    def __init__(self, Rs, AngDist, b_ell, bTF, r_ell=None, rTF=None, disc_fac=info['disc_fac'], **kwargs):
        # Setup FHT
        n = self.info['NNR']*self.info['resolution_factor']  # n must be same size as los? why?
        self.rht = FFTs.RadialFourierTransformHankel(lrange=[self.info['lmin'], self.info['lmax']], n=n, pad=self.info['pad'])

        self.beamTF = np.interp(self.rht.ell, b_ell, bTF)  # Load beam profile
        if r_ell is not None and rTF is not None:  # If resp is given, load
            self.beamrespTF = self.beamTF*np.interp(self.rht.ell, r_ell, rTF)

        self.los = np.geomspace(self.info['r_min'], self.info['r_max'], 200)  # line of sight integral, tested in 2.1
        self.rint = np.sqrt(self.los**2 + (self.rht.r[:,None])**2*AngDist**2)  # r values for LOS integration
        self.rs = np.geomspace(np.min(self.rint), np.max(self.rint), 100)  # r values for profile defined by limits of rints
        # self.rs = np.geomspace(np.radians(np.min(Rs)/60)*AngDist**2, np.radians(np.max(Rs)/60)*AngDist**2*disc_fac, 100)
        self.Rs = Rs
   
    def proj2D(self, prof_r):  # project along line of sight
        prof_int = interp1d(self.rs, prof_r, bounds_error=False, fill_value=0.0)(self.rint)  # interpret to integration rs
        prof_proj = 2*np.trapezoid(prof_int, x=self.los)  # Integrate over line of sight
        return prof_proj

    def beam_convolve(self, prof2D, beam, **kwargs):
        prof_ell_beam = self.rht.real2harm(prof2D)*beam  # Transform to harmonic space and colvolve with beam
        r_unpad, prof2D_beam = self.rht.unpad(self.rht.r, self.rht.harm2real(prof_ell_beam))  # Transform back and unpad
        return r_unpad.flatten(), prof2D_beam.flatten()
    
    def aperture_photometry(self, Rsprof, prof2D_beam, NNR=info['NNR'], disc_fac=info['disc_fac'], **kwargs):
        prof2D_beam = interp1d(Rsprof, prof2D_beam, kind="linear", bounds_error=False, fill_value=0.0)  # interpolate to aperture Rs
        def signal(R):
            dR = np.arctan(np.arctan(np.radians(R / 60.0))) / NNR  # fineness of R shells
            Rs_fine = (np.arange(NNR) + 1.0) * dR  # values of R shells
            return 2.0*np.pi*dR * np.sum(Rs_fine * prof2D_beam(Rs_fine))
        return np.array([2*signal(R) - signal(R*disc_fac) for R in self.Rs])