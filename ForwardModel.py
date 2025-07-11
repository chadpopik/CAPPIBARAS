"""
This file contains the meat of the forward model: everything that happens once you have your function for profile making and all the prep work (HODs, HMF, SMFs, etc.). Specifically, averaging the profile over mass and redshift, projecting it onto the sky and convolving it with the beam, and performing the aperture photometry.

The goal in all of these is to do as much precalculation as possible, as the only things that are changing as chains run is the value of certain parameters, and therefore anything that stays the same should NOT be run everytime. This is done by instead returning lambda functions of just the fit parameters.
"""

from Basics import *

import FFTs
from scipy.interpolate import interp1d  # Do we need this? can we just use normal numpy?



def weighting(galaxydist):
    gdist_norm = galaxydist/np.sum(galaxydist)
    return lambda Pths: np.sum(np.sum(gdist_norm * Pths, axis=1), axis=1)


def HODweighting(rs, zs, mshalo, Nc_func, Ns_func, uSat_func,
                 HMF, r200c_func, H_func, XH, **kwargs):
    ks, FFT_func = FFTs.mcfit_package(rs).FFT3D()
    rs_rev, IFFT_func = FFTs.mcfit_package(rs).IFFT1D()
    
    xs = rs[:, None, None]/r200c_func(zs[:, None], mshalo)
    # NOTE: These profiles also have some parameters that can be fit, but I'm not doing that here
    usk_m_z = FFT_func(uSat_func(xs)) 
    uck_m_z = np.ones(usk_m_z.shape)  # Set to one
    
    # Precalculating as much as possible
    yfac = (2+2*XH)/(3+5*XH)*(c.sigma_T/c.m_e/c.c**2).value * 4*np.pi*r200c_func(zs[:, None], mshalo)**3*((1+zs)**2/H_func(zs))[:, None]  # Converting Pth to y

    def HODave(Pths, params):
        Nc = Nc_func(mshalo, params)
        Ns = Ns_func(mshalo, params)
        ngal = np.trapz(np.trapz((Nc+Ns)*HMF, mshalo), zs)
        HODTerm = (Nc*uck_m_z + Ns*usk_m_z)/ngal

        dndzdm_norm = HODTerm*HMF/np.trapz(np.trapz(HODTerm*HMF, mshalo), zs)[:, None, None]
        yk_m_z = FFT_func(Pths)*yfac
        Pgy1h = np.trapz(np.trapz(yk_m_z*dndzdm_norm, mshalo), zs)

        yfacave = np.trapz(np.trapz(yfac*dndzdm_norm, mshalo), zs)
        return IFFT_func(Pgy1h/yfacave)

    return lambda Pths, params: HODave(Pths, params)


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


def aperture_photometry(thts, prof2D_beam, NNR, disc_fac):
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


def Pth_to_muK(XH, T_CMB, freq, **kwargs):
    factor = (c.sigma_T/c.m_e/c.c**2).cgs.value * (2+2*XH)/(3+5*XH) * fnu(freq, T_CMB) * T_CMB*1e6
    return lambda tSZ_sig: factor * tSZ_sig


def rho_to_muK(v_rms, XH, T_CMB, **kwargs):
    factor = v_rms * (c.sigma_T/c.m_p).cgs.value * (1+XH)/2 * T_CMB*1e6
    return lambda kSZ_sig: factor * kSZ_sig


def fnu(nu, T_cmb):
    x = (c.h * nu*u.GHz / (c.k_B * T_cmb*u.K)).decompose().value
    ans = x / np.tanh(x / 2.0) - 4.0
    return ans







# from hmvec
def C_yy_new(self,ells,zs,ks,Ppp,gzs,dndz=None,zmin=None,zmax=None):
    chis = self.comoving_radial_distance(gzs)
    hzs = self.h_of_z(gzs) # 1/Mpc
    Wz1s = 1/(1+gzs)
    Wz2s = 1/(1+gzs)
    # Convert to y units
    # 

def C_gy_new(self,ells,zs,ks,Pgp,gzs,gdndz=None,zmin=None,zmax=None):
    gzs = np.asarray(gzs)
    chis = self.comoving_radial_distance(gzs)
    hzs = self.h_of_z(gzs) # 1/Mpc
    nznorm = np.trapz(gdndz,gzs)
    term = (c.sigma_T/(c.m_e*c.c**2)).to(u.s**2/u.M_sun)*u.M_sun/u.s**2
    Wz1s = gdndz/nznorm
    Wz2s = 1/(1+gzs)

    return limber_integral(ells,zs,ks,Pgp,gzs,Wz1s,Wz2s,hzs,chis)

def C_gg_new(self,ells,zs,ks,Pgg,gzs,gdndz=None,zmin=None,zmax=None):
    gzs = np.asarray(gzs)
    chis = self.comoving_radial_distance(gzs)
    hzs = self.h_of_z(gzs) # 1/Mpc
    nznorm = np.trapz(gdndz,gzs)
    Wz1s = gdndz/nznorm
    Wz2s = gdndz/nznorm
    return limber_integral(ells,zs,ks,Pgg,gzs,Wz1s,Wz2s,hzs,chis)



# Limber Integral from hmvec
def limber_integral2(ells,zs,ks,Pzks,gzs,Wz1s,Wz2s,hzs,chis):
    """
    Get C(ell) = \int dz (H(z)/c) W1(z) W2(z) Pzks(z,k=ell/chi) / chis**2.
    ells: (nells,) multipoles looped over
    zs: redshifts (npzs,) corresponding to Pzks
    ks: comoving wavenumbers (nks,) corresponding to Pzks
    Pzks: (npzs,nks) power specrum
    gzs: (nzs,) corersponding to Wz1s, W2zs, Hzs and chis
    Wz1s: weight function (nzs,)
    Wz2s: weight function (nzs,)
    hzs: Hubble parameter (nzs,) in *1/Mpc* (e.g. camb.results.h_of_z(z))
    chis: comoving distances (nzs,)

    We interpolate P(z,k)
    """

    hzs = np.array(hzs).reshape(-1)
    Wz1s = np.array(Wz1s).reshape(-1)
    Wz2s = np.array(Wz2s).reshape(-1)
    chis = np.array(chis).reshape(-1)
    
    prefactor = hzs * Wz1s * Wz2s   / chis**2.
    zevals = gzs
    if zs.size>1:            
         f = interp2d(ks,zs,Pzks,bounds_error=True)     
    else:      
         f = interp1d(ks,Pzks[0],bounds_error=True)
    Cells = np.zeros(ells.shape)
    for i,ell in enumerate(ells):
        kevals = (ell+0.5)/chis
        if zs.size>1:
            # hack suggested in https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
            # to get around scipy.interpolate limitations
            interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zevals)[0]
        else:
            interpolated = f(kevals)
        if zevals.size==1: Cells[i] = interpolated * prefactor
        else: Cells[i] = np.trapz(interpolated*prefactor,zevals)
    return Cells



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





