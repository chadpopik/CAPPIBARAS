"""

- TODO 1: clean up the projection and aperture functions
- TODO 2: Maybe too much of th FFT mess is done in this file, it can be put into the FFT file?
- TODO 3: check if intergrating over zs is really necessary, maybe just use the medium z?
- TODO 4: Should i just drop the y conversion? like why is it in there? for rho it's incorrect, and for Pth it might be unneeded.
- TODO 5: Add miscentering?
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.interpolate import interp1d  # Do we need this? can we just use normal numpy?
import Models.FFTs as FFTs
from scipy.interpolate import RegularGridInterpolator



class BaseProjection:
    def fnu(self, freq, T_CMB=2.725, **kwargs):
        x = (c.h * freq*u.GHz / (c.k_B * T_CMB*u.K)).decompose().value
        return x / np.tanh(x / 2.0) - 4.0

    def Pth_to_uK(self, XH=0.76, **kwargs):
        factor = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH)*1e6 *(u.Mpc*u.sr).to(u.cm*u.arcmin**2)
        return lambda prof_r: self.aperture_photometry(*self.beam_convolve(prof_r, self.beamrespTF)) *factor

    def Pth_to_y(self, XH=0.76, **kwargs):
        factor = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH) *(u.Mpc*u.sr).to(u.cm*u.arcmin**2)
        return lambda prof_r: self.aperture_photometry(*self.beam_convolve(prof_r, self.beamTF)) *factor

    # def rho_to_uK(self, XH=0.76, v_rms=1.06e-3, T_CMB=2.725, **kwargs):
    #     factor = v_rms * (c.sigma_T/c.m_p).cgs.value * (1+XH)/2 * T_CMB*1e6 *(u.Mpc).to(u.cm)
    #     return lambda prof_r: self.aperture_photometry(*self.beam_convolve(prof_r, self.beamTF)) *factor
    
    def rho_to_uK(self, XH=0.76, v_rms=1.06e-3, T_CMB=2.725, **kwargs):
        factor = v_rms * (c.sigma_T/c.m_p).cgs * (1+XH)/2 * T_CMB*u.uK*1e6 * u.cm.to(u.Mpc)**2 * u.Mpc**2/u.cm**2 * u.g/u.Msun * u.Msun.to(u.g)
        return lambda prof_r: prof_r * factor



class Popik2025(BaseProjection):
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











# Old stuff

# Frequency dependence of the tSZ temperature anisotropy
def fnu(self, freq, T_CMB, **kwargs):
    x = (c.h * freq*u.GHz / (c.k_B * T_CMB*u.K)).decompose().value
    ans = x / np.tanh(x / 2.0) - 4.0
    return ans

def Pth_to_uK(rs, thetas, AngDist, beam_data, beam_ells, resp_data, resp_ells, XH=0.76, **kwargs):
    proj = project_Hankel(rs, thetas, AngDist, beam_data, beam_ells, resp_data, resp_ells)
    factor = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH)*1e6
    return lambda Pth3D: proj(Pth3D)*factor

def Pth_to_y(rs, thetas, AngDist, beam_data, beam_ells, XH=0.76, **kwargs):
    resp_ells, resp_data = beam_ells, np.ones_like(beam_ells)  # No response function for y
    proj = project_Hankel(rs, thetas, AngDist, beam_data, beam_ells, resp_data, resp_ells)
    factor = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH)
    return lambda Pth3D: proj(Pth3D)*factor

def rho_to_uK(rs, thetas, AngDist, beam_data, beam_ells, XH=0.76, v_rms=1.06e-3, T_CMB=2.725, **kwargs):
    resp_ells, resp_data = beam_ells, np.ones_like(beam_ells)  # No response function for kSZ
    proj = project_Hankel(rs, thetas, AngDist, beam_data, beam_ells, resp_data, resp_ells)
    factor = v_rms * (c.sigma_T/c.m_p).cgs.value * (1+XH)/2 * T_CMB*1e6
    return lambda rho3D: proj(rho3D)*factor


# Want this to return a lambda function in terms of r and Pth
# Should have angdist as a function that uses inputs from kwargs provider to be flexible with halo models
# TODO: check values of stuff like res_factor, NNR, disc_fac, etc
def project_Hankel(rs, thetas, AngDist, beam_data, beam_ells, resp_data, resp_ells,
                    resolution_factor=3.5, NNR=100, disc_fac=np.sqrt(2), sizeArcmin = 30.0, **kwargs):
    thta_max = np.arctan(np.arctan(sizeArcmin * np.pi/180.0/60.0/disc_fac))  # maximum map size to consider TODO: Does this really need a double arctan here?
    thta_smooth = thta_max * (np.arange(resolution_factor*NNR) + 1.0)/(resolution_factor*NNR)  # Equally spaced, finer 

    los = np.logspace(-3, 1, 200)  # line of sight to integrate over, NOTE: los arrays were tested in Popik 2025 and Moser 2023
    # NOTE: Testing was done to ensure that we can use the angular distance from the average redshift without a significant effect on the results
    rint = np.sqrt(los**2 + thta_smooth[:,None]**2*AngDist**2)

    # TODO: check value of pad
    rht = FFTs.RadialFourierTransformHankel(lrange=[170.0, 1.4e6], n=los.size, pad=100)  # n must be same size as los, lrange tested in Moser 2023

    beamTF = np.interp(rht.ell, beam_ells, beam_data)  # Load beam profile
    respTF = np.interp(rht.ell, resp_ells, resp_data)  # Load response
    beams = beamTF*respTF

    def project_convolve(prof3D):  # This has to be redone for every new profile, everything above is only done once
        prof2D = 2*np.trapezoid(interp1d(rs, prof3D, bounds_error=False, fill_value=0.0)(rint), x=los)  # Interpolate and integrate Pth over LOS
        lprofs = rht.real2harm(np.interp(rht.r, thta_smooth, prof2D))  # Interpolate and transform Pth to harmonic space
        rprofs = rht.harm2real(lprofs*beams)  # Convolve with beam and response and transform back to real space
        r_unpad, rprofs = rht.unpad(rht.r, rprofs)  # Unpad (removes points add on edges for smoothness)
        prof2D_beam = interp1d(r_unpad.flatten(), rprofs.flatten(), kind="linear", bounds_error=False, fill_value=0.0)  # Interpolate to whatever thetas are needed for aperture photometry
        return aperture_photometry(thetas, prof2D_beam, NNR, disc_fac)

    unitconv = (u.Mpc*u.sr).to(u.cm*u.arcmin**2)  # Put into cgs and arcmin^2 units
    return lambda prof3D: project_convolve(prof3D)*unitconv


def aperture_photometry(thts, # angular size of the measurements
                        prof2D_beam,
                        NNR, 
                        disc_fac):
    sig_all_p_beam = [] 
    for tht in thts:
        dtht_use = np.arctan(np.arctan(np.radians(tht / 60.0))) / NNR
        thta_use = (np.arange(NNR) + 1.0) * dtht_use
        sig_p = 2.0 * np.pi * dtht_use * np.sum(thta_use * prof2D_beam(thta_use))

        dtht2_use = np.arctan(np.arctan(np.radians(tht * disc_fac / 60.0))) / NNR
        thta2_use = (np.arange(NNR) + 1.0) * dtht2_use
        sig2_p = 2.0 * np.pi * dtht2_use * np.sum(thta2_use * prof2D_beam(thta2_use))

        sig_all_p_beam.append(sig_p-(sig2_p-sig_p))

    return np.array(sig_all_p_beam)








# The following two functions were written to see if keeping a z dependance during the projection functions would create much of a difference, instead of using an average z
# The answer was no, it barely changes anything at the cost of a significant time addition
def project_tsz_Hankel2(rs, zs, thetas, AngDistFunc, LumDistFunc, galaxydist,
                       beam,
                       resolution_factor=3.5, NNR=100, disc_fac=np.sqrt(2), sizeArcmin = 30.0, XH=0.76, **kwargs):
    thta_max = np.arctan(np.arctan(sizeArcmin * np.pi/180.0/60.0/disc_fac))  # maximum map size to consider
    thta_smooth = thta_max * (np.arange(resolution_factor*NNR) + 1.0)/(resolution_factor*NNR)  # Equally spaced, finer 

    dzs = np.concatenate([-np.geomspace(1e-8, 0.01, 100)[::-1],
                          np.geomspace(1e-8, 0.01, 100)])

    los = LumDistFunc(zs[:, None]) - LumDistFunc(zs[:, None]+dzs)
    
    dndz = np.sum(galaxydist, axis=0)
    dndz_norm = dndz/np.trapezoid(dndz, zs)
    # TODO: watch angdist here, if we're intergrating over the line of sight should we have to make angdist change?
    rint = np.sqrt(los**2 + thta_smooth[:, None, None]**2*AngDistFunc(zs[:, None]+dzs)**2)

    # TASK?: check value of pad
    rht = RadialFourierTransform(n=los.size, pad=100, lrange=[170.0, 1.4e6])  # n must be same size as los, lrange tested in Moser 2023
    beamTF = np.interp(rht.ell, beam.beam_ells, beam.beam_data)  # Load beam profile
    respTF = np.interp(rht.ell, beam.resp_ells, beam.resp_data)

    def project_convolve(Pths):  # This has to be redone for every new profile, everything above is only done once
        Pth_interps = [interp1d(rs, Pths[:, i], axis=0, bounds_error=False, fill_value=0.0) for i in range(Pths.shape[-1])]
        Pth_interp_z = np.array([Pth_interps[i](rint[:, i, :]) for i in range(Pths.shape[1])]).swapaxes(0, 1)
        
        Pth2D_z = np.trapezoid(-Pth_interp_z[..., :los.shape[-1]//2], los[:, :los.shape[-1]//2])+np.trapezoid(-Pth_interp_z[..., los.shape[-1]//2:], los[:, los.shape[-1]//2:])
        
        Pth2D = np.trapezoid(Pth2D_z*dndz_norm, zs) 
        
        # Interpolate and integrate Pth over LOS 
        lprofs = rht.real2harm(np.interp(rht.r, thta_smooth, Pth2D))  # Interpolate and transform Pth to harmonic space
        rprofs = rht.harm2real(lprofs*beamTF*respTF)  # Convolve with beam and response and transform back to real space
        r_unpad, rprofs = rht.unpad(rht.r, rprofs)  # Unpad (idk really what this means)
        Pth2D_beam = interp1d(r_unpad.flatten(), rprofs.flatten(), kind="linear", bounds_error=False, fill_value=0.0)  # Interpolate to whatever thetas are needed for aperture photometry
        return aperture_photometry(thetas, Pth2D_beam, NNR, disc_fac)

    # TASK: check units
    PthtoTtsz = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH) * (u.Mpc*u.sr).to(u.cm*u.arcmin**2)
    return lambda Pths: project_convolve(Pths)*PthtoTtsz




# def miscetner()
    # phis = np.linspace(0, 2*np.pi, 50)
    # R_mis = np.geomspace(2e-4, 2e1, 100)
    # rs_theta, rs_theta2  = thta_smooth[:, 0]*AngDis, thta2_smooth[:, 0]*AngDis
    # Rints = np.sqrt(R_mis[None, ..., None]**2+rs_theta[..., None, None]**2 \
    #                 +2*R_mis[None, ..., None]*rs_theta[..., None, None]*np.cos(phis))
    # Pth2D_R_Rmis = np.trapezoid(Pth2Dfunc(Rints, Pth_inter), phis, axis=-1)/(2*np.pi)

    # gamma = lambda r_mis: r_mis/tauRg**2 * np.exp(-r_mis/tauRg)
    # Pth2D_mis_R = np.trapezoid(gamma(R_mis) * Pth2D_R_Rmis, R_mis)

    # Pth2D_mis = (1-f_mis)*Pth2D+f_mis*Pth2D_mis_R