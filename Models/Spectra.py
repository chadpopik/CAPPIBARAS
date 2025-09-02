import numpy as np
import astropy.constants as c
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator


# General form
def Pgy1h(logM, hmf, H_y, H_g):
    return np.trapz(H_y*H_g*hmf, logM)

def H_y(FFT_func, Hz, XH, **kwargs):
    yfac = (c.sigma_T/c.m_e/c.c**2).to(u.s**2/u.M_sun).value * (2+2*XH)/(3+5*XH)
    prefac = c.c.to(u.km/u.s).value/Hz[:, None]*yfac
    return lambda Pth: FFT_func(prefac*Pth)

def H_g(Nc, Ns, uck, usk, logM, hmf, **kwargs):
    ngal = np.trapz((Nc+Ns)*hmf, logM)[:, None]
    return (Nc*uck + Ns*usk)/ngal
    
def HODweighting(zs, logmshalo, HMF, zdist, FFT_func, IFFT_func, H_g=None):
    if H_g is not None:
        # If the HOD is preset we can precalculate it once
        dndzdm_norm = H_g*HMF/np.trapz(np.trapz(H_g*HMF, logmshalo), zs)[:, None, None]
        return lambda Pth: IFFT_func(np.average(np.trapz(FFT_func(Pth)*dndzdm_norm, logmshalo), weights=zdist, axis=1))
    else:
        dndzdm_norm = lambda H_g: H_g*HMF/np.trapz(np.trapz(H_g*HMF, logmshalo), zs)[:, None, None]
        return lambda Pth, H_g: IFFT_func(np.average(np.trapz(FFT_func(Pth)*dndzdm_norm(H_g), logmshalo), weights=zdist, axis=1))



# Calculating C_ells
def limber(ks, zs, W_A, W_B, Hs, chis, ells):
    # Check the ell arrays to be within interpolation bounds
    ells_from_ks = ks[:, None]*chis
    if ells.min()<ells_from_ks.min(): raise ValueError(f"ell_min must be belaboveow {ells_from_ks.min()}")
    elif ells.max()>ells_from_ks.max(): raise ValueError(f"ell_max must be below {ells_from_ks.max()}")
    ks_from_ells = ells[:, None]/chis  # [n_ells]/[n_zs] > [n_ells, n_zs], precalculated
    
    # Make 2D interpolator and array of (k, z) points to put into it
    intp_points = np.stack((ks_from_ells, zs*np.ones(ks_from_ells.shape)), axis=-1)
    P_AB_int = lambda P_AB: RegularGridInterpolator((ks, zs), P_AB, bounds_error=False, fill_value=np.nan)(intp_points)

    # Precalculate the easy part
    intfac = Hs/chis**2/c.c.to(u.km/u.s).value * W_A * W_B
    return lambda P_AB: np.trapz(intfac*P_AB_int(P_AB))
    
def W_g(dNdz):  # Galaxy Kernel
    Ntot = np.trapz(dNdz)
    return dNdz/Ntot

def W_y(zs):  # Compton y kernel
    return 1/(1+zs)













# galaxy k-space kernel
def ug_k(rs, logM, rs200c, hmf, FFTf, Nc, Ns):
    xs = rs[:, None, None]/rs200c[None, ...]
    ngal = lambda p: np.trapz((self.Nc(logM, p)+self.Ns(logM, p))*hmf, logM)[:, None]  # mean number density of galaxies
    ugk = lambda p: (self.Nc(logM, p)*self.uck() + self.Ns(logM, p)*self.usk(xs, FFTf)(p)) / ngal(p)
    return lambda p={}: ugk(self.p0 | p)

# galaxy kernel squared, for autospectra
def ug_k2(self, rs, logM, rs200c, hmf, FFTf):
    xs = rs[:, None, None]/rs200c[None, ...]
    ngal = lambda p: np.trapz((self.Nc(logM, p)+self.Ns(logM, p))*hmf, logM)[:, None]  # mean number density of galaxies
    ugk2 = lambda p: (self.Ns(logM, p)**2*self.uck()**2 + 2*self.Ns(logM, p)*self.usk(xs, FFTf)(p)) / ngal(p)**2
    return lambda p={}: ugk2(self.p0 | p)

def Wg(dNdz, Hz, chi):  # galaxy kernel
    Ntot = np.trapz(dNdz)  # total number of galaxies
    phi = (1/Ntot) * dNdz  # normalized galaxy distribution
    return Hz/c.c * phi/chi**2
    
def Pgg(rs, zs, logmshalo, Nc_func, Ns_func, uSat_func,
                 HMF, r200c_func, H_func, XH, **kwargs):
    pass





# def HODweighting(rs, zs, logmshalo, Nc_func, Ns_func, uSat_func,
#                  HMF, r200c_func, H_func, XH, **kwargs):
#     ks, FFT_func = FFTs.mcfit_package(rs).FFT3D()
#     rs_rev, IFFT_func = FFTs.mcfit_package(rs).IFFT1D()

#     rs200c, Hs = r200c_func(zs, logmshalo), H_func(zs)
#     xs = rs[:, None, None]/rs200c
#     # NOTE: These profiles also have some parameters that can be fit, but I'm not doing that here
#     usk_m_z = FFT_func(uSat_func(xs)) 
#     uck_m_z = np.ones(usk_m_z.shape)  # Set to one
    
#     # Precalculating as much as possible
#     yfac = (2+2*XH)/(3+5*XH)*(c.sigma_T/c.m_e/c.c**2).value * 4*np.pi*rs200c**3*((1+zs)**2/Hs)[:, None]  # Converting Pth to y

#     def HODave(Pths, params):
#         Nc = Nc_func(logmshalo, params)
#         Ns = Ns_func(logmshalo, params)
#         ngal = np.trapz(np.trapz((Nc+Ns)*HMF, logmshalo), zs)
#         HODTerm = (Nc*uck_m_z + Ns*usk_m_z)/ngal

#         dndzdm_norm = HODTerm*HMF/np.trapz(np.trapz(HODTerm*HMF, logmshalo), zs)[:, None, None]
#         yk_m_z = FFT_func(Pths)*yfac
#         Pgy1h = np.trapz(np.trapz(yk_m_z*dndzdm_norm, logmshalo), zs)

#         yfacave = np.trapz(np.trapz(yfac*dndzdm_norm, logmshalo), zs)
#         return IFFT_func(Pgy1h/yfacave)

#     return lambda Pths, params: HODave(Pths, params)


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


def u_y(zs, mshalo, r200_func, dA_func):
    l200c = dA_func(zs)[:, None]/r200_func(zs, mshalo)
    prefac = 4*np.pi*r200_func(zs, mshalo)/l200c**2 * (c.sigma_T/c.m_e/c.c**2)
    return lambda Pek: prefac*Pek

def u_g(zs, mshalo, Nc, Ns, hmf):
    ng = np.trapz((Nc(mshalo)+Ns(mshalo))*hmf(zs, mshalo), np.log10(mshalo))
    
    


# Limber Integral from hmvec
# def limber_integral2(ells,zs,ks,Pzks,gzs,Wz1s,Wz2s,hzs,chis):
#     """
#     Get C(ell) = \int dz (H(z)/c) W1(z) W2(z) Pzks(z,k=ell/chi) / chis**2.
#     ells: (nells,) multipoles looped over
#     zs: redshifts (npzs,) corresponding to Pzks
#     ks: comoving wavenumbers (nks,) corresponding to Pzks
#     Pzks: (npzs,nks) power specrum
#     gzs: (nzs,) corersponding to Wz1s, W2zs, Hzs and chis
#     Wz1s: weight function (nzs,)
#     Wz2s: weight function (nzs,)
#     hzs: Hubble parameter (nzs,) in *1/Mpc* (e.g. camb.results.h_of_z(z))
#     chis: comoving distances (nzs,)

#     We interpolate P(z,k)
#     """

#     hzs = np.array(hzs).reshape(-1)
#     Wz1s = np.array(Wz1s).reshape(-1)
#     Wz2s = np.array(Wz2s).reshape(-1)
#     chis = np.array(chis).reshape(-1)
    
#     prefactor = hzs * Wz1s * Wz2s   / chis**2.
#     zevals = gzs
#     if zs.size>1:            
#          f = interp2d(ks,zs,Pzks,bounds_error=True)     
#     else:      
#          f = interp1d(ks,Pzks[0],bounds_error=True)
#     Cells = np.zeros(ells.shape)
#     for i,ell in enumerate(ells):
#         kevals = (ell+0.5)/chis
#         if zs.size>1:
#             # hack suggested in https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
#             # to get around scipy.interpolate limitations
#             interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zevals)[0]
#         else:
#             interpolated = f(kevals)
#         if zevals.size==1: Cells[i] = interpolated * prefactor
#         else: Cells[i] = np.trapz(interpolated*prefactor,zevals)
#     return Cells