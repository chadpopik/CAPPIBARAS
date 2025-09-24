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


# Frequency dependence of the tSZ temperature anisotropy
def fnu(freq, T_CMB, **kwargs):
    x = (c.h * freq*u.GHz / (c.k_B * T_CMB*u.K)).decompose().value
    ans = x / np.tanh(x / 2.0) - 4.0
    return ans

# Conversion between thermal pressure and muK for tSZ
def Pth_to_muK(XH, T_CMB, freq, **kwargs):
    factor = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH) * fnu(freq, T_CMB) * T_CMB*1e6
    return lambda tSZ_sig: factor * tSZ_sig

# Conversion of Pth to y
def Pth_to_y(XH, **kwargs):
    factor = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH)
    return lambda tSZ_sig: factor * tSZ_sig

# Conversion between gas density and muK for kSZ
def rho_to_muK(v_rms, XH, T_CMB, **kwargs):
    factor = v_rms * (c.sigma_T/c.m_p).cgs.value * (1+XH)/2 * T_CMB*1e6
    return lambda kSZ_sig: factor * kSZ_sig




# Want this to return a lambda function in terms of r and Pth
# Should have angdist as a function that uses inputs from kwargs provider to be flexible with halo models
# TODO: check values of stuff like res_factor, NNR, disc_fac, etc
def project_Hankel(rs, thetas, AngDist, beam_data, beam_ells, resp_data, resp_ells,
                       resolution_factor=3.5, NNR=100, disc_fac=np.sqrt(2), sizeArcmin = 30.0, **kwargs):
    # TODO: Does this really need a double arctan here?
    thta_max = np.arctan(np.arctan(sizeArcmin * np.pi/180.0/60.0/disc_fac))  # maximum map size to consider
    thta_smooth = thta_max * (np.arange(resolution_factor*NNR) + 1.0)/(resolution_factor*NNR)  # Equally spaced, finer 


    # NOTE: los arrays were tested in Popik 2025 and Moser 2023
    los = np.logspace(-3, 1, 200)  # line of sight to integrate over
    # NOTE: Testing was done to ensure that we can use the angular distance from the average redshift without a significant effect on the results
    rint = np.sqrt(los**2 + thta_smooth[:,None]**2*AngDist**2)

    # TODO: check value of pad
    rht = FFTs.RadialFourierTransformHankel(n=los.size, pad=100, lrange=[170.0, 1.4e6])  # n must be same size as los, lrange tested in Moser 2023

    beamTF = np.interp(rht.ell, beam_ells, beam_data)  # Load beam profile
    respTF = np.interp(rht.ell, resp_ells, resp_data)

    def project_convolve(prof3D):  # This has to be redone for every new profile, everything above is only done once
        prof2D = 2*np.trapz(interp1d(rs, prof3D, bounds_error=False, fill_value=0.0)(rint), x=los)  # Interpolate and integrate Pth over LOS 
        lprofs = rht.real2harm(np.interp(rht.r, thta_smooth, prof2D))  # Interpolate and transform Pth to harmonic space
        rprofs = rht.harm2real(lprofs*beamTF*respTF)  # Convolve with beam and response and transform back to real space
        r_unpad, rprofs = rht.unpad(rht.r, rprofs)  # Unpad (idk really what this means)
        prof2D_beam = interp1d(r_unpad.flatten(), rprofs.flatten(), kind="linear", bounds_error=False, fill_value=0.0)  # Interpolate to whatever thetas are needed for aperture photometry
        return aperture_photometry(thetas, prof2D_beam, NNR, disc_fac)

    unitconv = (u.Mpc*u.sr).to(u.cm*u.arcmin**2)        
    return lambda prof3D: project_convolve(prof3D)*unitconv


def aperture_photometry(thts, # angular size of the measurements
                        prof2D_beam, # function that 
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
def weighting2(ms, galaxydist): # This will return an array of size [n_rs, n_zs]
    dndm = np.sum(galaxydist, axis=1)
    dndm_norm = dndm[:, None]/np.trapz(dndm, ms)
    return lambda Pths: np.trapz(dndm_norm * Pths, ms, axis=1)


def project_tsz_Hankel2(rs, zs, thetas, AngDistFunc, LumDistFunc, galaxydist,
                       beam,
                       resolution_factor=3.5, NNR=100, disc_fac=np.sqrt(2), sizeArcmin = 30.0, XH=0.76, **kwargs):
    thta_max = np.arctan(np.arctan(sizeArcmin * np.pi/180.0/60.0/disc_fac))  # maximum map size to consider
    thta_smooth = thta_max * (np.arange(resolution_factor*NNR) + 1.0)/(resolution_factor*NNR)  # Equally spaced, finer 

    dzs = np.concatenate([-np.geomspace(1e-8, 0.01, 100)[::-1],
                          np.geomspace(1e-8, 0.01, 100)])

    los = LumDistFunc(zs[:, None]) - LumDistFunc(zs[:, None]+dzs)
    
    dndz = np.sum(galaxydist, axis=0)
    dndz_norm = dndz/np.trapz(dndz, zs)
    # TODO: watch angdist here, if we're intergrating over the line of sight should we have to make angdist change?
    rint = np.sqrt(los**2 + thta_smooth[:, None, None]**2*AngDistFunc(zs[:, None]+dzs)**2)

    # TASK?: check value of pad
    rht = RadialFourierTransform(n=los.size, pad=100, lrange=[170.0, 1.4e6])  # n must be same size as los, lrange tested in Moser 2023
    beamTF = np.interp(rht.ell, beam.beam_ells, beam.beam_data)  # Load beam profile
    respTF = np.interp(rht.ell, beam.resp_ells, beam.resp_data)

    def project_convolve(Pths):  # This has to be redone for every new profile, everything above is only done once
        Pth_interps = [interp1d(rs, Pths[:, i], axis=0, bounds_error=False, fill_value=0.0) for i in range(Pths.shape[-1])]
        Pth_interp_z = np.array([Pth_interps[i](rint[:, i, :]) for i in range(Pths.shape[1])]).swapaxes(0, 1)
        
        Pth2D_z = np.trapz(-Pth_interp_z[..., :los.shape[-1]//2], los[:, :los.shape[-1]//2])+np.trapz(-Pth_interp_z[..., los.shape[-1]//2:], los[:, los.shape[-1]//2:])
        
        Pth2D = np.trapz(Pth2D_z*dndz_norm, zs) 
        
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
    # Pth2D_R_Rmis = np.trapz(Pth2Dfunc(Rints, Pth_inter), phis, axis=-1)/(2*np.pi)

    # gamma = lambda r_mis: r_mis/tauRg**2 * np.exp(-r_mis/tauRg)
    # Pth2D_mis_R = np.trapz(gamma(R_mis) * Pth2D_R_Rmis, R_mis)

    # Pth2D_mis = (1-f_mis)*Pth2D+f_mis*Pth2D_mis_R