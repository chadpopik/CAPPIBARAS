"""
On the impacts of halo model implementations in Sunyaev-Zeldovich cross-correlation analyses

ui.adsabs.harvard.edu/abs/2025JCAP...10..051P
arxiv.org/pdf/2502.13291
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Popik2025")





"""Old implementation being phased out"""

from Models.Projections import BaseProjection
class Projection(BaseProjection):
    info = {}

    def __init__(self, 
                 Rs,  # values of R of measurement [arcmin]
                 AngDist,  # Mean Angular distance to halos [Mpc]
                 N_LOS=200,  # Resolution number of line of sight integral
                 N_RRHT=350,  # Resolution number of angular size
                 pad=100,  # number of points padded on sides of FHT ell/R range to help accuracy
                 **kwargs):

        self.Rs = Rs

        # Radial profile limits defined by measurement extent and beam smooth
        # TODO: these shouldn't be fixed, determine from measurement
        # Something like max measurements R * beam smoothing * disc_fac * dA(z)
        r_min = 1e-3*u.Mpc 
        r_max = 10*u.Mpc

        # Line of sight distance logspaced array of length N_LOS
        self.dLOS = np.geomspace(r_min, r_max, N_LOS)

        # FHT setup, defines angular scale (ell/R) logspaced array of length N_RRHT 
        self.rht = FFTs.RadialFourierTransformHankel(lrange=[np.floor(AngDist/r_max), np.ceil(AngDist/r_min)], n=N_RRHT, pad=pad)

        # Radial distance from the profile center as a function of line of sight and angular offset
        self.r3D = np.sqrt(self.dLOS**2 + (self.rht.r[:,None])**2*AngDist**2)  # r values for LOS integration

    def proj2D(self, rs):  # takes in function of R
        return lambda prof3D: 2*np.trapezoid(np.interp(self.r3D, rs, prof3D, left=0, right=0), x=self.dLOS)  # Integrate over line of sight

    # Function for convolving a beam/response with a 2D projected profile
    def beam_convolve(self, b_ell=None, bTF=None, r_ell=None, rTF=None, **kwargs):
        # Load beam profile and response (if given), and interpolate to RHT ells
        self.beamTF = np.interp(self.rht.ell, b_ell, bTF)
        if r_ell is not None and rTF is not None:
            self.beamTF = self.beamTF*np.interp(self.rht.ell, r_ell, rTF)

        def convolve(prof2d):
            prof_ell_beam = self.rht.real2harm(prof2d)*self.beamTF  # Transform to harmonic space and convolve with beam
            r_unpad, prof2D_beam = self.rht.unpad(self.rht.r, self.rht.harm2real(prof_ell_beam))  # Transform back and unpad
            return (r_unpad.flatten()*u.rad).to(u.arcmin), prof2D_beam.flatten()*prof2d.unit

        return lambda prof2D: convolve(prof2D)

    def aperture_photometry(self, f_disc=np.sqrt(2), N_RCAP=500, **kwargs):
        # Setup for CAP
        self.Rs_CAP = np.array([np.linspace(0, R, N_RCAP+1)[1:] for R in self.Rs])*u.arcmin  # values of R for circles
        self.Rs_CAP2 = np.array([np.linspace(0, f_disc*R, N_RCAP+1)[1:] for R in self.Rs])*u.arcmin  # values of R for outer disc
        self.infac_CAP = 2*np.pi*self.Rs[:, None]/N_RCAP *self.Rs_CAP  # prefactor for sum
        self.infac_CAP2 = 2*np.pi*f_disc*self.Rs[:, None]/N_RCAP *self.Rs_CAP2  # prefactor for sum, outer disc
        
        def CAP(Rsprof, prof2D_beam):
            sig = np.sum(self.infac_CAP * np.interp(self.Rs_CAP, Rsprof, prof2D_beam, right=0), axis=1)  # normal circle
            sig2 = np.sum(self.infac_CAP2 * np.interp(self.Rs_CAP2, Rsprof, prof2D_beam, right=0), axis=1)  # outer disc
            return (2*sig - sig2)

        return lambda Rsprof, prof2D_beam: CAP(Rsprof, prof2D_beam)
    
    def prof_to_signal(self, prof3D, rs):
        prof2D = self.proj2D(rs)(prof3D)  # project to 2D
        r_unpad, prof2D_beam = self.beam_convolve(prof2D)  # convolve with beam and response
        aper = self.aperture_photometry()
        return self.aperture_photometry()(r_unpad, prof2D_beam)  # aperture photometry  