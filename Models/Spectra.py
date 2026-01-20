"""
F
"""

# https://arxiv.org/pdf/2108.01601
# https://arxiv.org/pdf/1810.13423
# https://arxiv.org/pdf/1909.09102
# https://arxiv.org/pdf/2102.07701

import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.interpolate import RegularGridInterpolator

import Models.FFTs as FFTs
import Models.Studies as Studies
import Models.HODs as HODs
import Models.Data as Data




class BaseSpectra:
    def ngal(self, zs, logMs):  # mean galaxy density
        dndlogm = self.dndlogm(zs, logMs)  # setup
        return lambda Nc, Ns: np.trapz((Nc+Ns)*dndlogm, logMs*u.dex)
    
    def u_g(self, zs, logMs):  # galaxy overdensity fourier profile
        ngal = ngal(zs, logMs)  # setup
        return lambda Nc, Ns, nc, ns: (Nc*nc+Ns*ns)/ngal(Nc, Ns)

    def u_g2(self, zs, logMs):  # galaxy overdensity fourier profile for one-halom auto
        ngal = ngal(zs, logMs)  # setup
        return lambda Nc, Ns, nc, ns: (2*Nc*nc*Ns*ns + (Ns*ns)**2)/ngal**2
    
    def u_y(self, zs, **kwargs):  # compton y fourier profile
        yfac = (c.sigma_T/c.m_e/c.c**2).to(u.s**2/u.M_sun).value  # Conversion from P_e to y in cosmo units
        cgs_cosmo = (u.g/u.cm/u.s**2).to(u.M_sun/u.Mpc/u.s**2)  # Factor to convert Pe from CGS to cosmo units
        infac = yfac * cgs_cosmo  # combined pre-transform factors
        prefac = c.c.to(u.km/u.s).value/self.H(zs)  # post-transform factor, [Hz]=km/s/Mpc
        return lambda Pe_k: prefac*infac*Pe_k  # return lambda funciton to not recalculate above 

    def P1h_gg(self, zs, logMs):  # galaxy overdensity one-halo auto-spectra
        dndlogm, ug2 = self.dndlogm(zs, logMs), self.u_g2(zs, logMs)  # setup
        return lambda Nc, Ns, nc, ns: np.trapz(ug2(Nc, Ns, nc, ns)*dndlogm, logMs*u.dex)

    def P2h_gg(self, ks, zs, logM):  # galaxy overdensity one-halo auto-spectra
        u_g, intfac, Plin = u_g(zs, logM), self.dndlogm(zs, logM)*self.bh(zs, logM), self.Plin(ks, zs)  # setup
        return lambda Nc, Ns, nc, ns: Plin*np.trapz(u_g(Nc, Ns, nc, ns)*intfac, logM*u.dex)**2
    
    def P_gg(self, ks, zs, logM):  # galaxy overdensity total auto-spectra
        P1h_gg, P2h_gg = self.P1h_gg(zs, logM), self.P2h_gg(ks, zs, logM)  # setup
        return lambda Nc, Ns, nc, ns: P1h_gg(Nc, Ns, nc, ns)+P2h_gg(Nc, Ns, nc, ns)
    
    def P1h_gy(self, zs, logM):  # galaxy overdensity compton y one-halo cross-spectra
        u_g, u_y, dndlogm = u_g(zs, logM), u_y(zs), dndlogm(zs, logM)  # setup
        return lambda Nc, Ns, nc, ns, Pe: np.trapz(u_g(Nc, Ns, nc, ns)*u_y(Pe)*dndlogm, logM*u.dex)

    def P2h_gy(self, ks, zs, logM):  # galaxy overdensity compton y two-halo cross-spectra
        u_g, u_y, intfac, Plin = u_g(zs, logM), u_y(zs), self.dndlogm(zs, logM)*self.bh(zs, logM), self.Plin(ks, zs)  # setup
        return lambda Nc, Ns, nc, ns, Pe: Plin*np.trapz(u_g(Nc, Ns, nc, ns)*intfac, logM*u.dex)*np.trapz(u_y(Pe)*intfac, logM*u.dex)

    def P_gy(self, ks, zs, logM):  # galaxy overdensity total auto-spectra
        P1h_gy, P2h_gy = P1h_gy(zs, logM), P2h_gy(ks, zs, logM)  # setup
        return lambda Nc, Ns, nc, ns, Pe: self.P1h_gy(Nc, Ns, nc, ns, Pe)+self.P2h_gy(Nc, Ns, nc, ns, Pe)


    def C_AB(self, ells, ks, zs, W_A, W_B, beam, **kwargs):
        Pk_to_Pell = self.k_to_ell(ells, ks, self.chi(zs), zs)
        intfac = beam[:, None] * W_A * W_B * self.H(zs)/c.c.to(u.km/u.s).value/self.chi(zs)**2  # integrand factor
        return lambda P_AB: np.trapz(intfac*Pk_to_Pell(P_AB), zs)  # return lambda function to not recalculate above
    
    def W_g(self, dNdz, zs, **kwargs):  # Galaxy Kernel
        return dNdz/np.trapz(dNdz, zs)

    def W_y(self, zs, **kwargs):  # Compton y kernel
        return 1/(1+zs)

    def C_gg(self, ells, ks, zs, dNdz, **kwargs):
        C_gg_func = self.C_AB(ells, ks, zs, self.W_g(dNdz, zs), self.W_g(dNdz, zs), 1)
        return lambda P_gg: C_gg_func(P_gg)  # return lambda function to not recalculate above

    def C_gy(self, ells, ks, zs, dNdz, beam, **kwargs):
        C_gy_func = self.C_AB(ells, ks, zs, self.W_g(dNdz, zs), self.W_y(zs), beam)
        return lambda P_gy: C_gy_func(P_gy)  # return lambda function to not recalculate above

    def C_yy(self, ells, ks, zs, beam, **kwargs):
        C_yy_func = self.C_AB(ells, ks, zs, self.W_y(zs), self.W_y(zs), beam)
        return lambda P_yy: C_yy_func(P_yy)  # return lambda function to not recalculate above

    def D_ell(self, ells, C_ells):  # Get D_ell from C_ell
        return ells*(ells+1)*C_ells/2/np.pi

    def k_to_ell(self, ells, ks, zs, ell_func=lambda k, chi: k*chi-1/2):  # Interpolate a function in ks to corresponding ells (over all zs)
        ells_from_ks = ell_func(ks, self.chi(zs).value)  # corresponding ells from the ks of the function 
        if ells.min()<ells_from_ks.min() or ells.max()>ells_from_ks.max():   # ells can't correspond to ks outside the k range of the functions
            raise ValueError(f"ell must be in between {ells_from_ks.min()} and {ells_from_ks.max()}")
        ks_from_ells = (ells[:, None]+1/2)/self.chi(zs).value  # ks that correspond to input ells, 2D: [n_ells]/[n_zs] > [n_ells, n_zs]
        intp_points = np.stack((ks_from_ells, zs*np.ones(ks_from_ells.shape)), axis=-1)  # intrpn points of (k,z) that will get (ells_input, z)
        intp_func = lambda P_k: RegularGridInterpolator((ks, zs), P_k, bounds_error=False, fill_value=np.nan)  # interpolator to those points
        return lambda P_k: intp_func(P_k)(intp_points)




    # def Pth_k(): # TODO: working on it
    #     fft = FFTs.mcfit_package(ks=ks)  # setup FFT
    #     rs, FFT3D, IFFT3D = fft.rs, fft.FFT3D  # Define rs and FFT functions  
        
        
    #     efrac = (2+2*XH)/(3+5*XH)  # electron fraction
    #     fft = FFTs.mcfit_package(rs=rs)  # setup FFT
    #     ks, FFT3D, IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions






class Kusiak2022(BaseSpectra, HODs.Kusiak2022, Data.Kusiak2022):  # unWISE galaxies and Planck lensing (Kusiak+ 2023, arxiv.org/abs/2203.12583)

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)        
        logMs = np.linspace(7e8, 3.5e15, 20)/Studies.Kusiak2022().h
        zs = np.linspace(0.005, 4, 10)
        
    def d2VdzdOmega(self, zs):
        self.require(['H', 'chi'])
        return (c.c*self.chi(zs)**2/self.H(zs)).decompose()
    
    def ugl(self, ks, zs, logMs):
        self.require(['ells'])
        nsat, Pkell = self.ngal(zs, logMs), self.k_to_ell(self.ells[:, None, None], ks, zs)
        return lambda p: Pkell(nsat(p))
    
    def C_ij(self):  # Eq. 3
        return lambda C1h_ij, C2h_ij: C1h_ij+C2h_ij
    
    def C1h_ij(self, zs, logMs, **kwargs):  # Eq. 4
        self.require(['dndlogM'])
        intfactor = self.dndlogM(zs, logMs)*self.d2VdzdOmega(zs)
        return lambda u_i, u_j: np.trapz(np.trapz(intfactor*u_i*u_j, logMs), zs)

    def C2h_ij(self, ks, zs, logMsi, logMsj, **kwargs):  # Eq. 5
        self.require(['Plin', 'dndlogM', 'bh', 'ells'])
        Plin_k = self.k_to_ell(self.ells, ks[:, :, 0], zs[:, 0])(self.Plin(ks, zs)[:, :, 0])[:, :, None]
        intfac_i, intfac_j = Plin_k*self.d2VdzdOmega(zs)*self.dndlogM(zs, logMsi)*self.bh(zs, logMsi), self.dndlogM(zs, logMsj)*self.bh(zs, logMsj)
        return lambda u_i, u_j: np.trapz(np.trapz(intfac_i*u_i, logMsi)*np.trapz(intfac_j*u_j, logMsj), zs)

    def u_g(self, ks, zs, logMs, **kwargs):  # Eq. 11
        Wg, Nc, Ns, ugl, ngal = self.W_g(zs), self.Nc(logMs), self.Nc(logMs), self.ugl(ks, zs, logMs), self.ngal(zs, logMs)
        return lambda p: Wg/ngal(p) * (Nc(p)+Ns(p)*ugl)
    
    def ngal(self, zs, logMs, **kwargs):  # Eq. 12
        Nc, Ns, dndlogm = self.Nc(logMs), self.Nc(logMs), self.dndlogM(zs, logMs)
        return lambda p: np.trapz((Nc+Ns)*dndlogm, logMs)
    
    def W_g(self, zs):  # Eq 13 & 14
        self.require(['dNdz', 'H', 'chi'])
        phig = self.dNdz/np.trapz(self.dNdz, zs)
        return (self.H(zs)/c.c*phig/self.chi(zs)).decompose()

    def C1h_gg(self, ks, zs, logMs):  # Eq. 15
        C1hij, ug2 = self.C1h_ij(zs, logMs), self.u2_g(ks, zs, logMs)
        return lambda p: C1hij(ug2(p), 1)

    def u2_g(self, ks, zs, logMs):  # Eq. 16
        Wg, Ns, ugl, ngal = self.W_g(zs), self.Ns(logMs), self.ugl(ks, zs, logMs), self.ngal(zs, logMs)
        return lambda p: Wg**2/ngal(p)**2 * (Ns(p)**2*ugl(p)**2 + 2*Ns*ugl(p))

    def C2h_gg(self, ks, zs, logMs):  # Eq. 17
        C2hij, ug = self.C2h_ij(ks, zs, logMs, logMs), self.u_g(ks, zs, logMs)
        C2hgg = lambda ug: C2hij(ug)
        return lambda p: C2hgg(ug(p))
    
    def C_gg(self, ks, zs, logMs):
        C1hgg, C2hgg = self.C1h_gg(ks, zs, logMs), self.C2h_gg(ks, zs, logMs)
        return lambda p: C1hgg(p) + C2hgg(p)

    # def u_g(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, **kwargs):
    #     k_to_ell = self.Pk_to_Pell(ells, ks, chis, zs)
    #     W_g = self.W_g(Hz, chis, dNdz, zs)
    #     ngal = self.ngal(Nc, Ns, hmf, logM)
    #     return lambda p={}: W_g / ngal(p) * (Nc(p)+Ns(p)*k_to_ell(usk(p)))
    
    # def ngal(self, Nc, Ns, hmf, logM, zs, **kwargs):
    #     hmf
    #     return lambda p={}: np.trapz((Nc(p)+Ns(p))*hmf, logM)
        
    # def W_g(self, Hz, chis, dNdz, zs, **kwargs):
    #     phi_g = dNdz / np.trapz(dNdz, zs)
    #     return (Hz/c.c.to(u.km/u.s).value * phi_g/chis**2)[:, None]
    
    # def C1h_gg(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ells, ks, **kwargs):
    #     d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
    #     ug2 = self.ug2(Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs)
    #     return lambda p={}: np.trapz(d2V_dzdOmega*np.trapz(hmf*ug2(p), logM), zs)

    # def ug2(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs, **kwargs):
    #     k_to_ell = self.Pk_to_Pell(ells, ks, chis, zs)
    #     ul = lambda p: k_to_ell(usk(p))
    #     W_g = self.W_g(Hz, chis, dNdz, zs)
    #     ngal = self.ngal(Nc, Ns, hmf, logM)
    #     return lambda p={}: W_g**2/ngal(p)**2 * (Ns(p)**2*ul(p)**2 + 2*Ns(p)*ul(p))

    # def C2h_gg(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, Plin, bh, **kwargs):
    #     Plinl = self.uk_to_ul(ells, ks, chis, zs)(Plin)
    #     ug = self.u_g(ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks)
    #     d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
    #     return lambda p={}: np.trapz(d2V_dzdOmega*Plinl*np.trapz(bh*hmf*ug(p), logM)**2, zs)

    # def SN(self, area, dNdz, ells, zs, **kwargs):
    #     return area*(u.deg**2).to(u.sr)/np.trapz(dNdz, zs) *np.ones(ells.shape)



# class Popik2025(BaseSpectra):
#     def __init__(self, dndlogM, bh, Plin):


#     def P_AB(self, logM, **kwargs):  # 3D power spectrum
#         P_AB = lambda P1h_AB, P2h_AB: P1h_AB+P2h_AB
#         P1h_AB, P2h_AB = self.P1h_AB(logM, dndlogM), self.P2h_AB(logM, dndlogM, b_h, P_lin)
#         return lambda u_A, u_B: P_AB(P1h_AB(u_A, u_B), P2h_AB(u_A, u_B))

#     def P1h_AB(self, logM, dndlogM, **kwargs):  # one-halo 3D power spectrum
#         return lambda u_A, u_B: np.trapz(u_A*u_B*dndlogM, logM)

#     def P2h_AB(self, logM, dndlogM, b_h, P_lin, **kwargs):  # two-halo 3D power spectrum
#         intfac_A = dndlogM*b_h  # combined int factors
#         intfac_B = intfac_A *P_lin[..., None] # combined int factor, and throw Plin to not have to multiply it later
#         return lambda u_A, u_B: np.trapz(u_A*intfac_A, logM)*np.trapz(u_B*intfac_B, logM)

#     def u_g(self, logM, dndlogM):  # galaxy overdensity fourier profile
#         n_g = self.n_g(logM, dndlogM)
#         return lambda N_c, N_s, n_c, n_s: (N_c*n_c+N_s*n_s)/n_g(N_c, N_s)[:, None]

#     def u_g2(self, logM, dndlogM):  # one-halo auto-spectra galaxy overdensity profile
#         n_g = self.n_g(logM, dndlogM)
#         return lambda N_c, N_s, n_c, n_s: (2*N_c*n_c*N_s*n_s+N_s**2*n_s**2)/n_g(N_c, N_s)[:, None]**2

#     def n_g(self, logM, dndlogM):  # mean galaxy density
#         return lambda N_c, N_s: np.trapz((N_c+N_s)*dndlogM, logM)

#     def P1h_gg(self, logM, dndlogM, **kwargs):  # galaxy overdensity one-halo auto-spectra
#         P1h_gg = lambda u_g2: np.trapz(u_g2*dndlogM, logM)
#         u_g2 = self.u_g2(dndlogM, logM)
#         return lambda N_c, N_s, n_c, n_s: P1h_gg(u_g2(N_c, N_s, n_c, n_s))

#     def P2h_gg(self, logM, dndlogM, b_h, P_lin, **kwargs):  # galaxy overdensity two-halo auto-spectra
#         intfac = dndlogM*b_h  # combined int factors
#         P2h_gg = lambda u_g: P_lin*np.trapz(u_g*intfac, logM)**2
#         u_g = self.u_g(logM, dndlogM)
#         return lambda N_c, N_s, n_c, n_s: P2h_gg(u_g(N_c, N_s, n_c, n_s))

#     def P_gg(self, logM, dndlogM, b_h, P_lin, **kwargs):  # galaxy overdensity two-halo auto-spectra
#         P1h_gg, P2h_gg = self.P1h_gg(logM, dndlogM), self.P2h_gg(logM, dndlogM, b_h, P_lin)
#         return lambda N_c, N_s, n_c, n_s: P1h_gg(N_c, N_s, n_c, n_s)+P2h_gg(N_c, N_s, n_c, n_s)

#     def u_y(self, Hz, XH, **kwargs):  # compton-y fourier profile,
#         efrac = (2+2*XH)/(3+5*XH)  # electron fraction
#         yfac = (c.sigma_T/c.m_e/c.c**2).to(u.s**2/u.M_sun).value  # conversion factor for P_e to y [both cosmo units]
#         cgs_cosmo = (u.g/u.cm/u.s**2).to(u.M_sun/u.Mpc/u.s**2)  # conversion factor for Pth [CGS] to Pth [cosmo units]
#         infac = yfac * efrac * cgs_cosmo  # combined pre-transform factors
#         prefac = c.c.to(u.km/u.s).value/Hz[:, None]  # post-transform factor, [Hz]=km/s/Mpc
#         return lambda Pth: prefac*self.FFT3D(infac*Pth)  # return lambda function to not recalculate above   
    
#     def P_gy(self, logM, Hz, dndlogM, b_h, P_lin, XH=0.76, **kwargs):
#         u_g = self.u_g(logM, dndlogM)
#         u_y = self.u_y(Hz, XH)
#         P_gy = self.P_AB(logM, dndlogM, b_h, P_lin)
#         return lambda Pth, N_c, N_s, n_c, n_s: P_gy(u_y(Pth), u_g(N_c, N_s, n_c, n_s))
    
#     def P1h_gy(self, logM, Hz, dndlogM, XH=0.76, **kwargs):
#         u_g = self.u_g(logM, dndlogM)
#         u_y = self.u_y(Hz, XH)
#         P_gy = self.P1h_AB(logM, dndlogM)
#         return lambda Pth, N_c, N_s, n_c, n_s: P_gy(u_y(Pth), u_g(N_c, N_s, n_c, n_s))
    
#     def P2h_gy(self, logM, Hz, dndlogM, b_h, P_lin, XH=0.76, **kwargs):
#         u_g = self.u_g(logM, dndlogM)
#         u_y = self.u_y(Hz, XH)
#         P_gy = self.P2h_AB(logM, dndlogM, b_h, P_lin)
#         return lambda Pth, N_c, N_s, n_c, n_s: P_gy(u_y(Pth), u_g(N_c, N_s, n_c, n_s))


# class Popik2025(BaseSpectra):
#     def __init__(self, ells, zs, dNdz, chi, H, beam_ells=None, beam_data=None):
#         self.ells, self.zs, self.dNdz = ells, zs, dNdz
#         self.Hs, self.chis = chi(zs), H(zs)
        
        
#         self.Pk_to_Pell = self.k_to_ell(ells, self.ks, self.chis, zs, plushalf=True)  # setup k to ell interpolation
        
#         ks_est = (ells[:, None]+1/2)/self.chis
#         # TODO: define ks from rs or rs from ks?
#         kmin, kmax = np.min(ks_est), np.max(ks_est)
#         self.rs = np.geomspace(1/kmax, 1/kmin, 100)  # number of values is gonna be unknown
#         fft = FFTs.mcfit_package(rs=self.rs)
#         self.ks = fft.ks
#         self.FFT3D = fft.FFT3D
        
#         if beam_ells is not None and beam_ells is not None:
#             self.beam_data = np.interp(ells, beam_ells, beam_data)
#         else:
#             self.beam_data = 1

#     def C_AB(self, W_A, W_B, beam, **kwargs):
#         intfac = beam[:, None] * W_A * W_B * self.Hs/c.c.to(u.km/u.s).value/self.chis**2  # integrand factor
#         return lambda P_AB: np.trapz(intfac*self.Pk_to_Pell(P_AB), self.zs)  # return lambda function to not recalculate above
    
#     def C_gg(self, **kwargs):  # galaxy overdensity angular auto-spectra
#         C_AB = self.C_AB(self.W_g(), self.W_g(), np.ones(self.ells.shape))
#         ngaltot = np.trapz(self.dNdz, self.zs) *np.ones(self.ells.shape)
#         SN = lambda p: p['ASN']/ngaltot
#         return lambda P_gg, p={}: C_AB(P_gg)+SN(p | {'ASN':1})

#     def C_gy(self, **kwargs):  # galaxy overdensity compton-y cross-spectra
#         C_gy = self.C_AB(self.W_g(), self.W_y(), self.beam_data)
#         return lambda P_gy: C_gy(P_gy)

#     def P_AB(self, logM, dndlogM, b_h, P_lin, **kwargs):  # 3D power spectrum
#         P_AB = lambda P1h_AB, P2h_AB: P1h_AB+P2h_AB
#         P1h_AB, P2h_AB = self.P1h_AB(logM, dndlogM), self.P2h_AB(logM, dndlogM, b_h, P_lin)
#         return lambda u_A, u_B: P_AB(P1h_AB(u_A, u_B), P2h_AB(u_A, u_B))

#     def P1h_AB(self, logM, dndlogM, **kwargs):  # one-halo 3D power spectrum
#         return lambda u_A, u_B: np.trapz(u_A*u_B*dndlogM, logM)

#     def P2h_AB(self, logM, dndlogM, b_h, P_lin, **kwargs):  # two-halo 3D power spectrum
#         intfac_A = dndlogM*b_h  # combined int factors
#         intfac_B = intfac_A *P_lin[..., None] # combined int factor, and throw Plin to not have to multiply it later
#         return lambda u_A, u_B: np.trapz(u_A*intfac_A, logM)*np.trapz(u_B*intfac_B, logM)

#     def u_g(self, logM, dndlogM):  # galaxy overdensity fourier profile
#         n_g = self.n_g(logM, dndlogM)
#         return lambda N_c, N_s, n_c, n_s: (N_c*n_c+N_s*n_s)/n_g(N_c, N_s)[:, None]

#     def u_g2(self, logM, dndlogM):  # one-halo auto-spectra galaxy overdensity profile
#         n_g = self.n_g(logM, dndlogM)
#         return lambda N_c, N_s, n_c, n_s: (2*N_c*n_c*N_s*n_s+N_s**2*n_s**2)/n_g(N_c, N_s)[:, None]**2

#     def n_g(self, logM, dndlogM):  # mean galaxy density
#         return lambda N_c, N_s: np.trapz((N_c+N_s)*dndlogM, logM)

#     def P1h_gg(self, logM, dndlogM, **kwargs):  # galaxy overdensity one-halo auto-spectra
#         P1h_gg = lambda u_g2: np.trapz(u_g2*dndlogM, logM)
#         u_g2 = self.u_g2(dndlogM, logM)
#         return lambda N_c, N_s, n_c, n_s: P1h_gg(u_g2(N_c, N_s, n_c, n_s))

#     def P2h_gg(self, logM, dndlogM, b_h, P_lin, **kwargs):  # galaxy overdensity two-halo auto-spectra
#         intfac = dndlogM*b_h  # combined int factors
#         P2h_gg = lambda u_g: P_lin*np.trapz(u_g*intfac, logM)**2
#         u_g = self.u_g(logM, dndlogM)
#         return lambda N_c, N_s, n_c, n_s: P2h_gg(u_g(N_c, N_s, n_c, n_s))

#     def P_gg(self, logM, dndlogM, b_h, P_lin, **kwargs):  # galaxy overdensity two-halo auto-spectra
#         P1h_gg, P2h_gg = self.P1h_gg(logM, dndlogM), self.P2h_gg(logM, dndlogM, b_h, P_lin)
#         return lambda N_c, N_s, n_c, n_s: P1h_gg(N_c, N_s, n_c, n_s)+P2h_gg(N_c, N_s, n_c, n_s)

#     def u_y(self, Hz, XH, **kwargs):  # compton-y fourier profile,
#         efrac = (2+2*XH)/(3+5*XH)  # electron fraction
#         yfac = (c.sigma_T/c.m_e/c.c**2).to(u.s**2/u.M_sun).value  # conversion factor for P_e to y [both cosmo units]
#         cgs_cosmo = (u.g/u.cm/u.s**2).to(u.M_sun/u.Mpc/u.s**2)  # conversion factor for Pth [CGS] to Pth [cosmo units]
#         infac = yfac * efrac * cgs_cosmo  # combined pre-transform factors
#         prefac = c.c.to(u.km/u.s).value/Hz[:, None]  # post-transform factor, [Hz]=km/s/Mpc
#         return lambda Pth: prefac*self.FFT3D(infac*Pth)  # return lambda function to not recalculate above   
    
#     def P_gy(self, logM, Hz, dndlogM, b_h, P_lin, XH=0.76, **kwargs):
#         u_g = self.u_g(logM, dndlogM)
#         u_y = self.u_y(Hz, XH)
#         P_gy = self.P_AB(logM, dndlogM, b_h, P_lin)
#         return lambda Pth, N_c, N_s, n_c, n_s: P_gy(u_y(Pth), u_g(N_c, N_s, n_c, n_s))
    
#     def P1h_gy(self, logM, Hz, dndlogM, XH=0.76, **kwargs):
#         u_g = self.u_g(logM, dndlogM)
#         u_y = self.u_y(Hz, XH)
#         P_gy = self.P1h_AB(logM, dndlogM)
#         return lambda Pth, N_c, N_s, n_c, n_s: P_gy(u_y(Pth), u_g(N_c, N_s, n_c, n_s))
    
#     def P2h_gy(self, logM, Hz, dndlogM, b_h, P_lin, XH=0.76, **kwargs):
#         u_g = self.u_g(logM, dndlogM)
#         u_y = self.u_y(Hz, XH)
#         P_gy = self.P2h_AB(logM, dndlogM, b_h, P_lin)
#         return lambda Pth, N_c, N_s, n_c, n_s: P_gy(u_y(Pth), u_g(N_c, N_s, n_c, n_s))

    
# class Pandey2023(BaseSpectra):
#     def __init__(self):
#         pass
    
#     def u_y(self):
#         l200c = dA/r200c
#         prefactor = beam * 4*np.pi*r200c/l200c**2 * (c.sigma_T/c.m_e/c.c**2)


# class Kusiak2022(BaseSpectra):  # unWISE galaxies and Planck lensing (Kusiak+ 2023, arxiv.org/abs/2203.12583)
#     def __init__(self):
#         pass
    
#     def u_m(self, logM, rho0_m, r200c, c200c, lambda_trunc):  # Eq. 8
#         rho0 = 10**(logM)/rho0_m
#         return lambda lambda_trunc: rho0*self.NFW_k(rdel=r200c, cdel=c200c, lambda_trunc=lambda_trunc)
    
#     def u_g(self, W_g, **kwargs):  # Eq. 11
#         return lambda N_c, N_s, u_m, n_g: W_g / n_g * (N_c+N_s*u_m)
    
#     # def ngal(self, Nc, Ns, dndlogm, logM, **kwargs):  # Eq. 12
#     #     return np.trapz((Nc+Ns)*dndlogm, logM)
    
#     def C1h_gg(self, dndlogm, logms, d2VdzdOmega, zs):  # Eq. 15
#         intfactor = dndlogm*d2VdzdOmega
#         return lambda u2_g: np.trapz(np.trapz(intfactor*u2_g, logms), zs)
    
#     def u2_g(self, W_g, Hz, chis, dNdz, zs):  # Eq. 16
#         W_g = Hz/c * self.W_g(zs, dNdz)/chis**2  # Eq. 13 & 14
#         return lambda N_s, u_m, n_g: W_g**2/n_g**2 * (N_s**2*u_m**2 + 2*N_s*u_m)
    
#     def C2h_gg(self, dndlogm, logms, b_h, d2VdzdOmega, zs, Plin, ells, ks, chis):  # Eq. 17
#         Plin_k = self.Pk_to_Pell(ells, ks, chis, zs)(Plin)
#         intfac = Plin_k*d2VdzdOmega*dndlogm*b_h
#         return lambda u_g: np.trapz(np.trapz(intfac*u_g, logms)**2, zs)
    
#     def C_ij(self, C1h_ij, C2h_ij):  # Eq. 3
#         return C1h_ij+C2h_ij
    
#     def C1h_ij(self, dndlogm, logms, d2VdzdOmega, zs, **kwargs):  # Eq. 4
#         intfactor = dndlogm*d2VdzdOmega
#         return lambda u_i, u_j: np.trapz(np.trapz(intfactor*u_i*u_j, logms), zs)
    
#     def C2h_ij(self, dndlogm_i, dndlogm_j, logms_i, logms_j, b_h_i, b_h_j, d2VdzdOmega, zs, Plin, ells, ks, chis, **kwargs):  # Eq. 5
#         Plin_k = self.Pk_to_Pell(ells, ks, chis, zs)(Plin)
#         intfac_i, intfac_j = Plin_k*d2VdzdOmega*dndlogm_i*b_h_i, dndlogm_j*b_h_j
#         return lambda u_i, u_j: np.trapz(np.trapz(intfac_i*u_i, logms_i)*np.trapz(intfac_j*u_j, logms_j), zs)
    
    

    
    
    
    
    # def u_g(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, **kwargs):
    #     k_to_ell = self.Pk_to_Pell(ells, ks, chis, zs)
    #     W_g = self.W_g(Hz, chis, dNdz, zs)
    #     ngal = self.ngal(Nc, Ns, hmf, logM)
    #     return lambda p={}: W_g / ngal(p) * (Nc(p)+Ns(p)*k_to_ell(usk(p)))
    
    # def ngal(self, Nc, Ns, hmf, logM, zs, **kwargs):
    #     hmf
    #     return lambda p={}: np.trapz((Nc(p)+Ns(p))*hmf, logM)
        
    # def W_g(self, Hz, chis, dNdz, zs, **kwargs):
    #     phi_g = dNdz / np.trapz(dNdz, zs)
    #     return (Hz/c.c.to(u.km/u.s).value * phi_g/chis**2)[:, None]
    
    # def C1h_gg(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ells, ks, **kwargs):
    #     d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
    #     ug2 = self.ug2(Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs)
    #     return lambda p={}: np.trapz(d2V_dzdOmega*np.trapz(hmf*ug2(p), logM), zs)

    # def ug2(self, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, ells, ks, zs, **kwargs):
    #     k_to_ell = self.Pk_to_Pell(ells, ks, chis, zs)
    #     ul = lambda p: k_to_ell(usk(p))
    #     W_g = self.W_g(Hz, chis, dNdz, zs)
    #     ngal = self.ngal(Nc, Ns, hmf, logM)
    #     return lambda p={}: W_g**2/ngal(p)**2 * (Ns(p)**2*ul(p)**2 + 2*Ns(p)*ul(p))

    # def C2h_gg(self, ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks, Plin, bh, **kwargs):
    #     Plinl = self.uk_to_ul(ells, ks, chis, zs)(Plin)
    #     ug = self.u_g(ells, Nc, Ns, usk, hmf, logM, Hz, chis, dNdz, zs, ks)
    #     d2V_dzdOmega = c.c.to(u.km/u.s).value*chis**2/Hz
    #     return lambda p={}: np.trapz(d2V_dzdOmega*Plinl*np.trapz(bh*hmf*ug(p), logM)**2, zs)

    # def SN(self, area, dNdz, ells, zs, **kwargs):
    #     return area*(u.deg**2).to(u.sr)/np.trapz(dNdz, zs) *np.ones(ells.shape)

        


# class Kou2023(BaseSpectra):  # Planck and CMASS DR12 (Kou 2023, arxiv.org/abs/2211.07502)
#     def __init__(self, dndlogm):
#         pass

#     def C_SN(self, ells, area, dNdz, zs, **kwargs):  # Shot Noise, Eq. 9
#         frac = area/(4*np.pi*(180/np.pi)**2)
#         return 4*np.pi*(area/(4*np.pi*(180/np.pi)**2))/np.trapz(dNdz, zs) * np.ones(ells.shape)
    
#     def C_AB(self, W_A, W_B, chis, Hs, zs, ells, ks, **kwargs):  # Angular power spectra, Eq. 14
#         P_AB_k_to_ell = self.Pk_to_Pell(ells, ks, zs, chis, plushalf=False)  # P_AB(k) to P_AB(ell) interpolator
#         intfac = W_A * W_B * Hs/c.c.to(u.km/u.s).value/chis**2  # integrand factor
#         return lambda P_AB: np.trapz(intfac*P_AB(P_AB), zs)  # return a lambda function to not recalculate above
    
#     def W_g(self, dNdz, zs, **kwargs):  # Galaxy Kernel, Eq. 15
#         return dNdz/np.trapz(dNdz, zs)

#     def W_y(self, zs, **kwargs):  # Compton y kernel, Eq. 16
#         return 1/(1+zs)

#     def P1h(self, Hx, Hy, hmf, logM, **kwargs):  # one-halo power spectrum, Eq. 24
#         return np.trapz(Hx*Hy*hmf, logM)

#     def P2h(self, Hx, Hy, hmf, logM, bh, Plin, **kwargs):  # two-halo power spectrum, Eq. 25
#         return Plin*np.trapz(Hx*bh*hmf, logM)*np.trapz(Hy*bh*hmf, logM)

#     def H_c(self, Nc, Ns, logM, hmf, **kwargs):  # central function, Eq. 27
#         return Nc/self.n_gal(Nc, Ns, hmf, logM)

#     def H_s(self, Nc, Ns, usk, zs, logM, hmf, **kwargs):  # satellite function, Eq. 28
#         ngal = self.n_gal(zs, logM)
#         return lambda p: self.Ns(p)*usk(p)/ngal(p)[:, None]

#     def n_gal(self, zs, logM, **kwargs):  # comoving galaxy number density, Eq. 29
#         dndlogm = self.dndlogm(zs, logM)
#         return lambda p: np.trapz((self.Nc(p)+self.Ns(p))*dndlogm, logM)

#     def H_y(self, FFT_func, Hz, XH, **kwargs):  # Compton y function, Eq. 30
#         efrac = (2+2*XH)/(3+5*XH)  # electron fraction
#         yfac = (c.sigma_T/c.m_e/c.c**2).to(u.s**2/u.M_sun).value  # Conversion from P_e to y in cosmo units
#         cgs_cosmo = (u.g/u.cm/u.s**2).to(u.M_sun/u.Mpc/u.s**2)  # Factor to convert Pth from CGS to cosmo units
#         infac = yfac * efrac * cgs_cosmo  # combined pre-transform factors
#         prefac = c.c.to(u.km/u.s).value/Hz[:, None]  # post-transform factor, [Hz]=km/s/Mpc
#         return lambda Pth: prefac*FFT_func(infac*Pth)  # return lambda funciton to not recalculate above      

#     def P_gg_1h(self, Nc, Ns, usk, logM, hmf, **kwargs):
#         Hc = self.H_c(Nc, Ns, logM, hmf)
#         Hs = self.H_s(Nc, Ns, usk, logM, hmf)
#         P1h = self.P1h(hmf, logM)
#         return lambda p={}: 2*P1h(Hc(p), Hs(p)) + P1h(Hs(p), Hs(p))

#     def Pgg_2h(self, Nc, Ns, usk, logM, hmf, bh, Plin, **kwargs):
#         Hc = self.H_c(Nc, Ns, logM, hmf)
#         Hs = self.H_s(Nc, Ns, usk, logM, hmf)
#         P2h = self.P2h(hmf, logM, bh, Plin)
#         return lambda p={}: P2h(Hc(p), Hc(p)) + 2*P2h(Hc(p), Hs(p)) + P2h(Hs(p), Hs(p))

#     def Pgy_1h(self, Nc, Ns, usk, logM, hmf, FFT_func, Hz, XH, **kwargs):
#         Hc = self.H_c(Nc, Ns, logM, hmf)
#         Hs = self.H_s(Nc, Ns, usk, logM, hmf)
#         Hy = self.H_y(FFT_func, Hz, XH)
#         P1h = self.P1h(hmf, logM)
#         return lambda Pth, p={}: P1h(Hc(p), Hy(Pth)) + P1h(Hs(p), Hy(Pth))

#     def Pgy_2h(self, Nc, Ns, usk, logM, hmf, FFT_func, Hz, XH, bh, Plin, **kwargs):
#         Hc = self.H_c(Nc, Ns, logM, hmf)
#         Hs = self.H_s(Nc, Ns, usk, logM, hmf)
#         Hy = self.H_y(FFT_func, Hz, XH)
#         P2h = self.P2h(hmf, logM, bh, Plin)
#         return lambda Pth, p={}: P2h(Hc(p), Hy(Pth)) + P2h(Hs(p), Hy(Pth))

#     def Cgg1h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, **kwargs):
#         Pgg_1h = self.Pgg_1h(Nc, Ns, usk, logM, hmf)
#         W_g = self.W_g(dNdz, zs)  # Galaxy Kernel, Eq. 15
#         Cl = self.C_ell(ells, ks, zs, W_g, W_g, chis, Hs)
#         return lambda p={}: Cl(Pgg_1h(p))
    
#     def Cgg2h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, bh, Plin, **kwargs):
#         Pgg_2h = self.Pgg_2h(Nc, Ns, usk, logM, hmf, bh, Plin)
#         W_g = self.W_g(dNdz, zs)  # Galaxy Kernel, Eq. 15
#         Cl = self.C_ell(ells, ks, zs, W_g, W_g, chis, Hs)
#         return lambda p={}: Cl(Pgg_2h(p))
    
#     def Cgy1h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, FFT_func, beam_ells, beam_data, XH, **kwargs):
#         Pgy_1h = self.Pgy_1h(Nc, Ns, usk, logM, hmf, FFT_func, Hs, XH)
#         W_g = self.W_g(dNdz, zs)  # Galaxy Kernel, Eq. 15
#         Cl = self.C_ell(ells, ks, zs, W_g, self.W_y(zs), chis, Hs)
#         beam = np.interp(ells, beam_ells, beam_data)
#         return lambda Pth, p={}: beam*Cl(Pgy_1h(Pth, p))
    
#     def Cgy2h(self, ells, ks, zs, chis, Hs, dNdz, Nc, Ns, usk, logM, hmf, FFT_func, bh, Plin, beam_ells, beam_data, XH, **kwargs):
#         Pgy_2h = self.Pgy_2h(Nc, Ns, usk, logM, hmf, FFT_func, Hs, XH, bh, Plin)
#         W_g = self.W_g(dNdz, zs)  # Galaxy Kernel, Eq. 15
#         Cl = self.C_ell(ells, ks, zs, W_g, self.W_y(zs), chis, Hs)
#         beam = np.interp(ells, beam_ells, beam_data)
#         return lambda Pth, p={}: beam*Cl(Pgy_2h(Pth, p))
    
#     def Cgg(self):
#         C_SN = 4*np.pi*(area/(4*np.pi*(180/np.pi)**2))/np.trapz(dNdz, zs) * np.ones(ells.shape)













# from hmvec
# def C_yy_new(self,ells,zs,ks,Ppp,gzs,dndz=None,zmin=None,zmax=None):
#     chis = self.comoving_radial_distance(gzs)
#     hzs = self.h_of_z(gzs) # 1/Mpc
#     Wz1s = 1/(1+gzs)
#     Wz2s = 1/(1+gzs)
#     # Convert to y units
#     # 

# def C_gy_new(self,ells,zs,ks,Pgp,gzs,gdndz=None,zmin=None,zmax=None):
#     gzs = np.asarray(gzs)
#     chis = self.comoving_radial_distance(gzs)
#     hzs = self.h_of_z(gzs) # 1/Mpc
#     nznorm = np.trapz(gdndz,gzs)
#     term = (c.sigma_T/(c.m_e*c.c**2)).to(u.s**2/u.M_sun)*u.M_sun/u.s**2
#     Wz1s = gdndz/nznorm
#     Wz2s = 1/(1+gzs)

#     return limber_integral(ells,zs,ks,Pgp,gzs,Wz1s,Wz2s,hzs,chis)

# def C_gg_new(self,ells,zs,ks,Pgg,gzs,gdndz=None,zmin=None,zmax=None):
#     gzs = np.asarray(gzs)
#     chis = self.comoving_radial_distance(gzs)
#     hzs = self.h_of_z(gzs) # 1/Mpc
#     nznorm = np.trapz(gdndz,gzs)
#     Wz1s = gdndz/nznorm
#     Wz2s = gdndz/nznorm
#     return limber_integral(ells,zs,ks,Pgg,gzs,Wz1s,Wz2s,hzs,chis)


# def u_y(zs, mshalo, r200_func, dA_func):
#     l200c = dA_func(zs)[:, None]/r200_func(zs, mshalo)
#     prefac = 4*np.pi*r200_func(zs, mshalo)/l200c**2 * (c.sigma_T/c.m_e/c.c**2)
#     return lambda Pek: prefac*Pek

# def u_g(zs, mshalo, Nc, Ns, hmf):
#     ng = np.trapz((Nc(mshalo)+Ns(mshalo))*hmf(zs, mshalo), np.log10(mshalo))


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