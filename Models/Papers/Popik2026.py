"""
Some Future Paper
"""

from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Popik2026")

from Models.Codes import mcfit




class CAP():
    def __init__(self, 
                 Rs,  # values of R of measurement [arcmin]
                 f_ring=np.sqrt(2), # determines size of outer ring
                 
                 N_RCAP=200, # determines resolution of disk/ring signal sum, set by convergence
                 **kwargs):
        
        # NOTE: we define new radius values here because we need an array R values that stops exactly at the disk/ring sizes we desire, and has sufficient resolution for the integral no matter what disk/ring size we go out to. the output of the beam convolution isn't necessarily going to have the latter, and defintely won't have the former, so we interpolate to these new arrays.
        self.Rs_disk = np.array([np.linspace(0, R, N_RCAP+1)[1:] for R in Rs])*u.arcmin  # values of R for circles
        self.Rs_ring = np.array([np.linspace(0, f_ring*R, N_RCAP+1)[1:] for R in Rs])*u.arcmin  # values of R for outer disc
        
        # Precalculate some integrand factors
        self.intfac_disk = 2*np.pi*Rs[:, None]/N_RCAP *self.Rs_disk  # prefactor for sum
        self.intfac_ring = 2*np.pi*f_ring*Rs[:, None]/N_RCAP *self.Rs_ring  # prefactor for sum, outer disc

    def disk_ring_CAP(self, Rsprof, prof2D_beam):
        signal_disk = np.sum(self.intfac_disk * np.interp(self.Rs_disk, Rsprof, prof2D_beam, right=0), axis=1)  # normal circle
        signal_ring = np.sum(self.intfac_ring * np.interp(self.Rs_ring, Rsprof, prof2D_beam, right=0), axis=1)  # outer disc
        return (2*signal_disk - signal_ring)
    



    
from Models.Codes import mopc
class BeamConvolution():
    info = {}

    def __init__(self, 
                 RMax,  # Maximum angular size of the output, in arcmin
                 RMin,  # Minimum angular size of the output, in arcmin
                 b_ell, # beam profile ells
                 bTF, # beam profile
                 r_ell=None, # response profile ells
                 rTF=None, # response profile
                 
                 f_beam=2.5*1.6, # factor*Rmaxout that setsmaximum angular size for the beam convolution
                 # essentially how much further out we should check in the profiles to account for the beam smoothing things outside Rmax into the Rmax radius
                 # Should be dependent on the FWHM of the beam and its shape
                 
                 # Following values can based just on convergence
                 RNUM_rht = 200,  # Angular size resolution
                 rht_pad = 250,  # number of points padded on sides of FHT ell/R range to help accuracy
                 **kwargs):

        # Maximum angular size in the beam convolution
        RMax_beam = f_beam*RMax
        RMin_beam = RMin
        
        # some code for maybe calculating f_beam
        # (u.rad/model.beam_ells[np.searchsorted(np.cumsum(model.beam)/np.sum(model.beam), 0.13)]).to(u.arcmin)

        # FHT setup, defines angular scale (ell/R) logspaced array of length N_RRHT 
        self.rht = mopc.RadialFourierTransformHankel(lrange=[np.floor(1/RMax_beam.to(u.rad).value), np.ceil(1/RMin_beam.to(u.rad).value)], n=RNUM_rht, pad=rht_pad) 
        
        # We don't actually need have a profile for the pad values, they're just there to help the FFT be less weird on the edges
        self.R_beam_unpad = (self.rht.r[rht_pad:-rht_pad] *u.rad).to(u.arcmin)
        self.R_beam_pad = (self.rht.r *u.rad).to(u.arcmin)
        self.logRpad = np.log10(self.R_beam_pad.value)
        self.logRunpad = np.log10(self.R_beam_unpad.value)

        # Load beam profile and response (if given), and interpolate to RHT ells
        self.beamTF = np.interp(self.rht.ell, b_ell, bTF)
        if r_ell is not None and rTF is not None:
            self.beamTF = self.beamTF*np.interp(self.rht.ell, r_ell, rTF)
            
        self.r_unpad_test, self.beamTF_unpad = self.rht.unpad(self.rht.r, self.beamTF)
            
        
    # Function for convolving a beam/response with a 2D projected profile
    def beam_convolve(self, prof2d):
        # Extrapolate the padded values with a simple loglinear interp
        prof2d_interp = 10**(scipy.interpolate.interp1d(self.logRunpad, np.log10(prof2d.value), kind='linear', fill_value='extrapolate')(self.logRpad))
        
        # Transform to harmonic space and convolve with beam
        prof_ell_beam = self.rht.real2harm(prof2d_interp)*self.beamTF
        
        # Transform back and unpad
        r_unpad, prof2D_beam = self.rht.unpad(self.rht.r, self.rht.harm2real(prof_ell_beam))
        return (r_unpad.flatten()*u.rad).to(u.arcmin), prof2D_beam.flatten()*prof2d.unit

    
    
class Projection():
    def __init__(self, 
                 Rs_beam,  # values of R of we'll be using from the beam [arcmin]
                 zMean,  # Mean redshift of hhalos [Mpc]
                 cosmology,  # cosmology

                 # The following values are set by just seeing if the integral converges
                 rLOSMax = 10*u.Mpc,  # maximum los of sight radius
                 rLOSMin = 1e-3*u.Mpc,  # minimum los of sight radius
                 rLOSNum = 200,  # Resolution number of line of sight integral
                 **kwargs):

        self.cosmology=cosmology
        # Line of sight distance logspaced array of length
        self.rLOS = np.geomspace(rLOSMin, rLOSMax, rLOSNum)
        
        # Array of of perpindicular distance determined by the angular distance values needed to input into the beam convolution process
        AngDist = self.cosmology.dA(zMean)
        Rs_beam_Mpc = Rs_beam.to(u.rad).value*AngDist
        
        # 2D array, one dimension in LOS direction the other in sky direction
        self.r3D = np.sqrt(self.rLOS**2 + Rs_beam_Mpc[:,None]**2)
    
    def proj2D(self, rs, prof3D):
        # interpolate the profile to the high resolution two dimensional array we use for the intergration.
        # we set a left and right value here, but they shouldn't be needed as the extent of the rs should reach across all the r3D values we need.
        prof2D_interp = np.interp(self.r3D, rs, prof3D, left=prof3D[0].value, right=0)
        
        # Integrate over line of sight
        return 2*np.trapezoid(prof2D_interp, x=self.rLOS)
    
    def to_y(self, **kwargs):
        return (c.sigma_T/c.m_e/c.c**2).cgs * (2+2*self.cosmology.XH)/(3+5*self.cosmology.XH) * u.g/u.Msun * u.Msun.to(u.g)
    
    def to_TtSZ(self, nu, **kwargs):
        x = (c.h * nu / (c.k_B * self.cosmology.T_CMB)).decompose().value
        return self.to_y() * (x / np.tanh(x / 2.0) - 4.0 ) * self.cosmology.T_CMB*u.uK*1e6
    
    def to_TkSZ(self, v_rms, **kwargs):
        return (v_rms/c.c).decompose() * (c.sigma_T/c.m_p).cgs * (1+self.cosmology.XH)/2 * self.cosmology.T_CMB*u.uK*1e6 * u.cm.to(u.Mpc)**2 * u.Mpc**2/u.cm**2 * u.g/u.Msun * u.Msun.to(u.g)
    
    
    
class HaloProfile():
    def __init__(self, r, z, logM200c, halomodel, **kwargs):
        self.__dict__.update(locals())
        
        # Format array into proper dimensions
        self.r, self.z, self.logM200c = r[:, None, None], z[:, None], logM200c
        
        # Precalculate some things
        self.M200c = 10**logM200c*u.Msun
        self.rhoc = halomodel.rhoc(self.z)
        self.r200c = (3*self.M200c/(4*np.pi*200*self.rhoc))**(1/3)
        self.x = self.r/self.r200c
        self.Fb = halomodel.Omega_b/halomodel.Omega_m   
        
    def zdep(self, alpha_z=0):
        return (1+self.z)**alpha_z
    
    def mdep(self, alpha_m=0):
        return (self.M200c/(1e14*u.Msun))**alpha_m

    def PL(self, A0, alpha_z=0, alpha_m=0):
        zterm = (1+self.z)**alpha_z
        mterm = (self.M200c/(1e14*u.Msun))**alpha_m
        return A0*zterm*mterm


class Density1h(HaloProfile):
    params = {
        # Density Parameters
        'rho0': 1.99, 'rho0_z': -0.66, 'rho0_m': 0.29,  # LOG amplitude
        'xc_rho': 0.6, 'xc_rho_z': 0, 'xc_rho_m': 0,   # Core radius
        'beta_rho': 2.6, 'beta_rho_z': -0.025, 'beta_rho_m': 0.04,  # Outer slope
        'gamma_rho': -0.2, 'gamma_rho_z': 0, 'gamma_rho_m': 0,  # Inner Slope
        'alpha_rho': 1, 'alpha_rho_z': 0.19, 'alpha_rho_m': -0.03,  # Intermediate Slope
    }

    def __init__(self, r, z, logM200c, halomodel, **kwargs):
        super().__init__(r, z, logM200c, halomodel, **kwargs)
        
        self.p0 = self.params

        # scale factor for density profile, in units of Msun/Mpc^3
        self.rho_scale = self.Fb*200*self.rhoc.to(u.Msun/u.Mpc**3)                


    def onehalo(self, pdict={}, **kwargs): 
        p = self.p0 | pdict | kwargs
        
        # rho0=self.PL(p['rho0'], p['rho0_z'], p['rho0_m'])
        # xc=self.PL(p['xc_rho'], p['xc_rho_z'], p['xc_rho_m'])
        # gamma=self.PL(p['gamma_rho'], p['gamma_rho_z'], p['gamma_rho_m'])
        # alpha=self.PL(p['alpha_rho'], p['alpha_rho_z'], p['alpha_rho_m'])
        # beta=self.PL(p['beta_rho'], p['beta_rho_z'], p['beta_rho_m'])
        
        rho0 = p['rho0']
        xc = p['xc_rho']
        gamma = p['gamma_rho']
        alpha = p['alpha_rho']
        beta = p['beta_rho']
        
        return self.rho_scale * rho0 * (self.x/xc)**gamma * (1+(self.x/xc)**alpha)**(-(beta-gamma)/alpha)
        # return self.rho_scale * rho0 * (self.x/xc)**gamma * (1+(self.x/xc)**alpha)**(-beta)
        
        
    
class Pressure1h(HaloProfile):
    models = {}
    params = {
        # Pressure Parameters
        'P0': 2.0, 'P0_z': -0.758, 'P0_m': 0.154,       # Amplitude
        'alpha_P': 0.8, 'alpha_P_z': 0, 'alpha_P_m': 0,  # Intermediate slope
        'beta_P': 2.6, 'beta_P_z': 0.415, 'beta_P_m': 0.0393,  # Outer slope
        'gamma_P': -0.3, 'gamma_P_z': 0, 'gamma_P_m': 0, # Inner Slope
        'xc_P': 0.497, 'xc_P_z': 0.731, 'xc_P_m': -0.00865,   # Core Radius
    }

    def __init__(self, r, z, logM200c, halomodel, **kwargs):
        super().__init__(r, z, logM200c, halomodel, **kwargs)
        
        self.p0 = self.params
        
        self.P_scale = self.Fb*(c.G*(self.M200c)*200*self.rhoc/(2*self.r200c)).to(u.Msun/u.Mpc/u.s**2)  # scale factor for density profile, in units of Msun/Mpc^3

    def onehalo(self, pdict={}, **kwargs): 
        p = self.p0 | pdict | kwargs
        
        # P0=self.PL(p['P0'], p['P0_z'], p['P0_m'])
        # xc=self.PL(p['xc_P'], p['xc_P_z'], p['xc_P_m'])
        # gamma=self.PL(p['gamma_P'], p['gamma_P_z'], p['gamma_P_m'])
        # alpha=self.PL(p['alpha_P'], p['alpha_P_z'], p['alpha_P_m'])
        # beta=self.PL(p['beta_P'], p['beta_P_z'], p['beta_P_m'])
        
        P0 = p['P0']
        xc = p['xc_P']
        gamma = p['gamma_P']
        alpha = p['alpha_P']
        beta = p['beta_P']
        
        return self.P_scale * P0 * (self.x/xc)**gamma * (1+(self.x/xc)**alpha)**(-beta)
    
    
from Models.Codes import mcfit
class HaloProfile2h(HaloProfile):
    def __init__(self, r, z, logM200c, halomodel,
                 logM200c_2h = np.linspace(10, 15, 10), **kwargs):
        self.__dict__.update(locals())
        self.M200c_2h = 10**logM200c_2h*u.Msun
        
        super().__init__(r, z, logM200c, halomodel, **kwargs)
        
        fft = mcfit.DHT(rs=r)  # setup FFT
        self.k, self.FFT3D, self.IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions

        self.prefac_2h = self.halomodel.bh(z, self.logM200c)*self.halomodel.Plin(self.k, z)  # collect factors outside int
        self.intfac_2h = self.halomodel.dndlogm(z, self.logM200c_2h)*self.halomodel.bh(z, self.logM200c_2h)  # collect factors inside int: uses M200h instead of other

    def twohalofunc(self, prof1h):
        P2h_k = self.prefac_2h*(np.trapezoid(self.FFT3D(prof1h)*self.intfac_2h,self.logM200c_2h*u.dex))[..., None]  # integrate of 2h mass range
        return self.IFFT3D(P2h_k) * prof1h.unit
    
    def total(self, pdict={}, **kwargs):
        p = self.p0 | pdict | kwargs
        return self.onehalo(p) + self.twohalo(p)
    

class Density(Density1h, HaloProfile2h):
    params2h = {'A_2h_rho': 1,}    # 2h amplitude
    def __init__(self, r, z, logM200c, halomodel, twohalofix="fixed", **kwargs):
        self.__dict__.update(locals())
        Density1h.__init__(self, r, z, logM200c, halomodel, **kwargs)
        HaloProfile2h.__init__(self, r, z, logM200c, halomodel, **kwargs)
        
        self.p0 = self.params | self.params2h
        
        # density profile defined over the twohalo integration mass range
        self.density_2hMarray = Density1h(r=r, z=z, logM200c=self.logM200c_2h, halomodel=halomodel)
        
        self.fixed2h = self.twohalofunc(self.density_2hMarray.onehalo())
        
        if self.twohalofix=='fixed': self.twohalo = self.twohalo_fixed
        elif self.twohalofix=='vary': self.twohalo = self.twohalo_vary

    def twohalo_vary(self, pdict={}, **kwargs):  # two-halo density component
        p = self.p0 | pdict | kwargs
        return p['A_2h_rho']*self.twohalofunc(self.density_2hMarray.onehalo(p))
    
    def twohalo_fixed(self, pdict={}, **kwargs):  # two-halo density component
        p = self.p0 | pdict | kwargs
        return p['A_2h_rho']*self.fixed2h

    
class Pressure(Pressure1h, HaloProfile2h):
    params2h = {'A_2h_P': 1,}    # 2h amplitude
    def __init__(self, r, z, logM200c, halomodel, twohalofix="fixed", **kwargs):
        self.__dict__.update(locals())
        Pressure1h.__init__(self, r, z, logM200c, halomodel, **kwargs)
        HaloProfile2h.__init__(self, r, z, logM200c, halomodel, **kwargs)
        
        self.p0 = self.params | self.params2h
        
        # density profile defined over the twohalo integration mass range
        self.pressure_2hMarray = Pressure1h(r=r, z=z, logM200c=self.logM200c_2h, halomodel=halomodel)
        
        self.fixed2h = self.twohalofunc(self.pressure_2hMarray.onehalo())
        
        if self.twohalofix=='fixed': self.twohalo = self.twohalo_fixed
        elif self.twohalofix=='vary': self.twohalo = self.twohalo_vary

    def twohalo_vary(self, pdict={}, **kwargs):  # two-halo density component
        p = self.p0 | pdict | kwargs
        return p['A_2h_P']*self.twohalofunc(self.pressure_2hMarray.onehalo(p))
    
    def twohalo_fixed(self, pdict={}, **kwargs):  # two-halo density component
        p = self.p0 | pdict | kwargs
        return p['A_2h_P']*self.fixed2h
    
    
    
    
    
    
    
class Spectra():  
    def __init__(self, rs, z, logMh, Ncen, Nsat, ucen, usat, dndlogMh, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

        # Define FFTs and get ks from the FFT functions
        self.fft = mcfit.FFT(rs)
        self.ks = self.fft.ks/u.Mpc
        self.FFT3D, self.IFFT2D = self.fft.FFT3D, self.fft.IFFT2D
        
        self.z = z
        self.logMh = logMh
        self.dndlogMh = dndlogMh(z=self.z, logM=self.logMh)
        
        self.Ncen, self.Nsat, self.ucen, self.usat = Ncen, Nsat, ucen, usat

        try:
            ngal = self.ngal(self.Ncen, self.Nsat)
            Hg = self.Hg(self.Ncen, self.ucen, self.Nsat, self.usat, ngal)
            Hg_norm = Hg/np.trapezoid(Hg*self.dndlogMh, logMh)[..., None]
            self.intfac = Hg_norm*self.dndlogMh
        except:
            pass

            
    def ngal(self, Ncen, Nsat): # total galaxy number
        return np.trapezoid((Ncen+Nsat)*self.dndlogMh, self.logMh)
    
    def Hg(self, Ncen, ucen, Nsat, usat, ngal):# HOD cross-spectra function
        return (Ncen*ucen + Nsat*usat)/ngal[None, :, None]
        
    def HODweighting(self, prof,p={}, **kwargs):
        if hasattr(self, "intfac"):
            intfac = self.intfac
        else:
            ngal = self.ngal(self.Ncen(p), self.Nsat(p))
            Hg = self.Hg(self.Ncen(p), self.ucen(p), self.Nsat(p), self.usat(p), ngal)
            Hg_norm = Hg/np.trapezoid(Hg*self.dndlogMh, self.logMh)[..., None]
            intfac = Hg_norm*self.dndlogMh

        return self.IFFT2D(np.trapezoid(self.FFT3D(prof)*intfac, self.logMh)) *prof.unit # take mass average

    
    


class Spectra_old():  
    def __init__(self, rs, z, logMh, dNdz, Ncen, Nsat, ucen, usat, dndlogMh, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

        # Define FFTs and get ks from the FFT functions
        self.fft = mcfit.FFT(rs)
        self.ks = self.fft.ks/u.Mpc
        self.FFT3D, self.IFFT1D = self.fft.FFT3D, self.fft.IFFT1D
        
        self.z = z
        self.logMh = logMh
        self.dNdz = dNdz
        self.dndlogMh = dndlogMh(z=self.z, logM=self.logMh)
        
        self.Ncen, self.Nsat, self.ucen, self.usat = Ncen, Nsat, ucen, usat

        infac = self.dndlogMh*dNdz[:, None]  # combined mass/z distributions
        try:
            ngal = self.ngal(self.Ncen, self.Nsat)
            Hg = self.Hg(self.Ncen, self.ucen, self.Nsat, self.usat, ngal)
            Hg_norm = Hg/np.trapezoid(np.trapezoid(Hg*infac, logMh), z)[:, None, None]
            self.intfac = Hg_norm*infac
        except:
            pass

            
    def ngal(self, Ncen, Nsat): # total galaxy number
        return np.trapezoid((Ncen+Nsat)*self.dndlogMh, self.logMh)
    
    def Hg(self, Ncen, ucen, Nsat, usat, ngal):# HOD cross-spectra function
        return (Ncen*ucen + Nsat*usat)/ngal[None, :, None]
        
    def HODweighting(self, prof,p={}, **kwargs):
        if hasattr(self, "intfac"):
            intfac = self.intfac
        else:
            ngal = self.ngal(self.Ncen(p), self.Nsat(p))
            Hg = self.Hg(self.Ncen(p), self.ucen(p), self.Nsat(p), self.usat(p), ngal)
            Hg_norm = Hg/np.trapezoid(np.trapezoid(Hg*self.infac, self.logMh), self.z)[:, None, None]
            intfac = Hg_norm*self.infac

        aveprof = np.trapezoid(np.trapezoid(self.FFT3D(prof)*intfac, self.logMh), self.z) # take mass/redshift average

        return lambda prof, p={}: self.IFFT1D(aveprof)*prof.unit
        
    # def HODweighting(self, HOD, hmod, targets, **kwargs):
    #     logM = targets.logMh
    #     zs = targets.z

    #     Nc, Ns = HOD.Ncen(logM), HOD.Nsat(logM)
    #     nc, ns = HOD.ncen(self.ks, zs, logM), HOD.nsat(self.ks, zs, logM)

    #     dndlogm = hmod.dndlogm(zs, logM)  # calculate halo mass function
    #     # dndlogm[:, (targets.logM<12) | (targets.logM>16)] = 0
    #     infac = dndlogm*targets.dNdz[:, None]  # combined mass/z distributions

    #     ngal = lambda p: np.trapezoid((Nc(p)+Ns(p))*dndlogm, logM)  # total galaxy number
    #     Hg = lambda p: (Nc(p)*nc(p) + Ns(p)*ns(p))/ngal(p)[None, :, None]  # HOD cross-spectra function
    #     # Hg = lambda p: (Nc(p)[None, None, :] + Ns(p)[None, None, :])/ngal(p)[:, None]  # HOD cross-spectra function
    #     Hg_norm = lambda p: Hg(p)/np.trapezoid(np.trapezoid(Hg(p)*infac, logM), zs)[:, None, None]  # normalized galaxy distribution
    #     intfac0 = Hg_norm({})*infac  # combine default HOD galaxy dist into integrand factor
    #     intfac = lambda p: Hg_norm(p)*infac if p!={} else intfac0  # recalculate integrand factor if HOD galaxy dist is being fit
    
    #     aveprof = lambda prof, p: np.trapezoid(np.trapezoid(self.FFT3D(prof)*intfac(p), logM), zs) # take mass/redshift average

    #     return lambda prof, p={}: self.IFFT1D(aveprof(prof, p))*prof.unit
    
    
    
"""Old implementation being phased out"""




# class Measurement(Study):  # TODO: In progress
#     path = f"{DATA_PATH}/Results"
#     subs = {
#     'zbin': ['z1', 'z2', 'z3', 'z4'],
#     'deproj' : ['Base', 'cib', 'cib_cibdBeta', 'cib_cibdBeta_cibdT', 'cib_cibdT'],
#     'TCIB': ['10.7', '24.0'],
#     'beta': ['1.0', '1.2', '1.4', '1.6', '1.7', '1.8', '2.0'],
#     }

#     def __init__(self, inputsdict, **inputvars):        
#         self.setup(inputsdict | inputvars)
        
#         self.require(['deproj'])
#         if self.deproj!='cib_cibdBeta_cibdT': # Add values
#             self.subs['TCIB'] = ['10.7']
#         if self.deproj!='cib_cibdBeta': # Add values
#             self.subs['TCIB'] = self.subs['TCIB']+['1.15', '1.30', '1.35', '1.45']

#         if self.deproj=='Base': self.require(['deproj'])
#         else: self.require(['deproj', 'TCIB', 'beta'])


#         self.get_meas()

#         # Taking properties from Zhou
#         Zhou = Zhou2023({'zbin':self.zbin,'sample':'main','hemisphere':'combined'})
#         for val in ['zs', 'dNdz', 'dndz', 'dz', 'area', 'logMhMean', 'zMean']:
#             setattr(self, val, getattr(Zhou, val))

#     def get_meas(self):
#         with h5py.File(f"{self.path}/ACTDR6DESILRG_Spectra_testnew.h5", 'r') as f:
#             self.ell = f['ell'][()]
#             self.Cgg_data = f[f'gxg/{self.zbin[-1]}'][()]
#             self.Cgy_data = f[f'gxy/{self.zbin[-1]}/{self.deproj}/{self.TCIB}/{self.beta}'][()]
#             self.Cyy_data = f[f'yxy/{self.deproj}/{self.TCIB}/{self.beta}'][()]

#             self.Cgy_err = np.abs(self.Cgy_data)/10000*self.ell  # TODO: placeholder
            
            
# from CAPPIBARAS.Models.OldModules.Profiles import BaseProfile
# from Models.Papers import Amodeo2021, Vikram2017
# class HaloProfile_old(BaseProfile, Amodeo2021.Study):
#     PLparams = {'zPL': ['logrho0', 'xc_k', 'beta_k', 'gamma_k', 'alpha_k', 'P0', 'alpha_t', 'beta_t', 'xc_t', 'gamma_t'],
#               'mPL': ['logrho0', 'xc_k', 'beta_k', 'gamma_k', 'alpha_k', 'P0', 'alpha_t', 'beta_t', 'xc_t', 'gamma_t']}  # pres/dens profile model
#     def __init__(self, inputsdict={}, **inputvars):
#         self.setup(inputsdict | inputvars, model=True)
        
#     def r200c(self, z, logM200c):
#         M200c = 10**logM200c*u.Msun
#         return (3*M200c/(4*np.pi*200*self.rhoc(z)))**(1/3)

#     def PL(self, z, logM200c, pname):
#         if not hasattr(self, 'zPL'): setattr(self, 'zPL', [])
#         if not hasattr(self, 'mPL'): setattr(self, 'mPL', [])
#         zterm = lambda alphaz: (1+z)**alphaz
#         mterm = lambda alpham: (10**logM200c/1e14)**alpham
#         zterm0 = zterm(self.p0[f"{pname}_z"])
#         mterm0 = mterm(self.p0[f"{pname}_m"])

#         func = lambda A0: A0
        
#         if pname in self.zPL: func1 = lambda A0, alphaz: func(A0=A0)*zterm(alphaz)
#         else: func1 = lambda A0, alphaz: func(A0)*zterm0
        
#         if pname in self.mPL: func2 = lambda A0, alphaz, alpham: func1(A0=A0, alphaz=alphaz)*mterm(alpham)
#         else: func2 = lambda A0, alphaz, alpham: func1(A0=A0, alphaz=alphaz)*mterm0

#         return func2
    
#     def prof2h(self, r, z, logM200c): 
#         dndlogm = lambda z, logM200c: self.dndlogm(z, logM200c)
#         bh = lambda z, logM200c: self.bh(z, logM200c)
#         Plin = lambda k, z: self.Plin(k, z)
        
#         V17 = Vikram2017.Profiles(dndlogm=dndlogm, bh=bh, Plin=Plin)
#         logM200c_2h = np.linspace(10, 15, 10)
#         lin2h = V17.twohalo(r, z, logM200c, logM200c_2h)  # linear two-halo calculation
#         return lambda prof, p={}: lin2h(prof(r, z, logM200c_2h)(p))
    
#     def total(self, r, z, logM200c, units='cosmo'):
#         onehalo, twohalo = self.onehalo(r, z, logM200c, units), self.twohalo(r, z, logM200c, units)
#         return lambda p={}: onehalo(self.p0 | p) + twohalo(self.p0 | p)
    
    
# class Pressure_old(HaloProfile):
#     models = {}
#     params = {
#         # Pressure Parameters
#         'P0': 2.0, 'P0_z': -0.758, 'P0_m': 0.154,       # Amplitude
#         'alpha_t': 0.8, 'alpha_t_z': 0, 'alpha_t_m': 0,  # Intermediate slope
#         'beta_t': 2.6, 'beta_t_z': 0.415, 'beta_t_m': 0.0393,  # Outer slope
#         'gamma_t': -0.3, 'gamma_t_z': 0, 'gamma_t_m': 0, # Inner Slope
#         'xc_t': 0.497, 'xc_t_z': 0.731, 'xc_t_m': -0.00865,   # Core Radius
#         'A2h_t': 1,   # 2h amplitude
#     }
#     def __init__(self, inputsdict={}, **inputvars):
#         self.setup(inputsdict | inputvars, model=True)
        
#     def PGNFW(self, x, P0, xc, gamma, alpha, beta):  # Eq 17
#         return P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)
        
#     def P200c(self, z, logM200c, units='cosmo'): 
#         P200c = c.G*(10**logM200c*u.Msun)*200*self.rhoc(z)/(2*self.r200c(z, logM200c))
#         return self.Fb*P200c.to(self.units('pres', units))
    
#     def onehalo(self, r, z, logM200c, units='cosmo'):  
#         r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
#         P200c = self.P200c(z, logM200c, units)
#         x = r*u.Mpc/self.r200c(z, logM200c)
        
#         gamma=self.PL(z, logM200c, 'gamma_t')
#         alpha=self.PL(z, logM200c, 'alpha_t') 
#         P0=self.PL(z, logM200c, 'P0')
#         xc=self.PL(z, logM200c, 'xc_t')
#         beta=self.PL(z, logM200c, 'beta_t')
        
#         PGNFW = lambda p: self.PGNFW(x, 
#             gamma=gamma(p['gamma_t'], p['gamma_t_z'], p['gamma_t_m']), 
#             alpha=alpha(p['alpha_t'], p['alpha_t_z'], p['alpha_t_m']), 
#             P0=P0(p['P0'], p['P0_z'], p['P0_m']), 
#             xc=xc(p['xc_t'], p['xc_t_z'], p['xc_t_m']), 
#             beta=beta(p['beta_t'], p['beta_t_z'], p['beta_t_m']))
#         return lambda p={}: P200c*PGNFW(self.p0 | p)
    
#     def twohalo(self, r, z, logM200c, units='cosmo'):  # two-halo pressure component
#         twohalocalc = self.prof2h(r, z, logM200c)
#         return lambda p={}: (self.p0 | p)['A2h_t']*twohalocalc(self.onehalo, p ).to(self.units('pres', units))
    
    
# class Density_old(HaloProfile):
#     models = {}
#     params = {
#         # Density Parameters
#         'rho0': 1.99, 'rho0_z': -0.66, 'rho0_m': 0.29,  # LOG amplitude
#         'xc_k': 0.6, 'xc_k_z': 0, 'xc_k_m': 0,   # Core radius
#         'beta_k': 2.6, 'beta_k_z': -0.025, 'beta_k_m': 0.04,  # Outer slope
#         'gamma_k': -0.2, 'gamma_k_z': 0, 'gamma_k_m': 0,  # Inner Slope
#         'alpha_k': 1, 'alpha_k_z': 0.19, 'alpha_k_m': -0.03,  # Intermediate Slope
#         'A2h_k': 1,    # 2h amplitude
#     }

#     def __init__(self, inputsdict={}, **inputvars):
#         self.setup(inputsdict | inputvars, model=True)
        
#     def pGNFW(self, x, rho0, xc, gamma, alpha, beta):
#         return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta-gamma)/alpha)

#     def pc(self, z, units='cosmo'):
#         return self.Fb*200*self.rhoc(z).to(self.units('dens', units))  # prefactor and units

#     def onehalo(self, r, z, logM200c, units='cosmo'): 
#         r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
#         pc = self.pc(z, units)
#         x = r*u.Mpc/self.r200c(z, logM200c)

#         gamma=self.PL(z, logM200c, 'gamma_k')
#         alpha=self.PL(z, logM200c, 'alpha_k') 
#         rho0=self.PL(z, logM200c, 'rho0')
#         xc=self.PL(z, logM200c, 'xc_k')
#         beta=self.PL(z, logM200c, 'beta_k')

#         pGNFW = lambda p: self.pGNFW(x, 
#             gamma=gamma(p['gamma_k'], p['gamma_k_z'], p['gamma_k_m']), 
#             alpha=alpha(p['alpha_k'], p['alpha_k_z'], p['alpha_k_m']), 
#             rho0=rho0(p['rho0'], p['rho0_z'], p['rho0_m']), 
#             xc=xc(p['xc_k'], p['xc_k_z'], p['xc_k_m']), 
#             beta=beta(p['beta_k'], p['beta_k_z'], p['beta_k_m']))
#         return lambda p={}: pc*pGNFW(self.p0 | p)
    
#     def twohalo(self, r, z, logM200c, units='cosmo'):  # two-halo density component
#         twohalocalc = self.prof2h(r, z, logM200c)
#         return lambda p={}: (self.p0 | p)['A2h_k']*twohalocalc(self.onehalo, p | {par[:-3]: p[par] for par in p if '_2h' in par}).to(self.units('dens', units))
    
    

    

# class Profile(BaseProfile, Amodeo2021.Study):  #
#     models = {}
#     params = {
#         # Density Parameters
#         'logrho0': 2.6, 'logrho0_z': -0.66, 'logrho0_m': 0.29,  # LOG amplitude
#         'xc_k': 0.6, 'xc_k_z': 0, 'xc_k_m': 0,   # Core radius
#         'beta_k': 2.6, 'beta_k_z': -0.025, 'beta_k_m': 0.04,  # Outer slope
#         'gamma_k': -0.2, 'gamma_k_z': 0, 'gamma_k_m': 0,  # Inner Slope
#         'alpha_k': 1, 'alpha_k_z': 0.19, 'alpha_k_m': -0.03,  # Intermediate Slope
#         'A2h_k': 1,    # 2h amplitude

#         # Pressure Parameters
#         'P0': 2.0, 'P0_z': -0.758, 'P0_m': 0.154,       # Amplitude
#         'alpha_t': 0.8, 'alpha_t_z': 0, 'alpha_t_m': 0,  # Intermediate slope
#         'beta_t': 2.6, 'beta_t_z': 0.415, 'beta_t_m': 0.0393,  # Outer slope
#         'gamma_t': -0.3, 'gamma_t_z': 0, 'gamma_t_m': 0, # Inner Slope
#         'xc_t': 0.497, 'xc_t_z': 0.731, 'xc_t_m': -0.00865,   # Core Radius
#         'A2h_t': 1,   # 2h amplitude
#     }
#     PLparams = {'zPL': ['logrho0', 'xc_k', 'beta_k', 'gamma_k', 'alpha_k', 'P0', 'alpha_t', 'beta_t', 'xc_t', 'gamma_t'],
#               'mPL': ['logrho0', 'xc_k', 'beta_k', 'gamma_k', 'alpha_k', 'P0', 'alpha_t', 'beta_t', 'xc_t', 'gamma_t']}  # pres/dens profile model
#     def __init__(self, inputsdict={}, **inputvars):
#         self.setup(inputsdict | inputvars, model=True)

#     def PL(self, z, logM200c, pname):
#         if not hasattr(self, 'zPL'): setattr(self, 'zPL', [])
#         if not hasattr(self, 'mPL'): setattr(self, 'mPL', [])
#         zterm = lambda alphaz: (1+z)**alphaz
#         mterm = lambda alpham: (10**logM200c/1e14)**alpham
#         zterm0 = zterm(self.p0[f"{pname}_z"])
#         mterm0 = mterm(self.p0[f"{pname}_m"])

#         func = lambda A0: A0
        
#         if pname in self.zPL: func1 = lambda A0, alphaz: func(A0=A0)*zterm(alphaz)
#         else: func1 = lambda A0, alphaz: func(A0)*zterm0
        
#         if pname in self.mPL: func2 = lambda A0, alphaz, alpham: func1(A0=A0, alphaz=alphaz)*mterm(alpham)
#         else: func2 = lambda A0, alphaz, alpham: func1(A0=A0, alphaz=alphaz)*mterm0

#         return func2

#     def pGNFW(self, x, rho0, xc, gamma, alpha, beta):
#         return rho0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-(beta-gamma)/alpha)

#     def pc(self, z, units='cosmo'):
#         return self.Fb*self.rhoc(z).to(self.units('dens', units))  # prefactor and units

#     def Density1h(self, r, z, logM200c, units='cosmo'): 
#         r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
#         pc = self.pc(z, units)
#         x = r*u.Mpc/self.r200c(z, logM200c)

#         gamma=self.PL(z, logM200c, 'gamma_k')
#         alpha=self.PL(z, logM200c, 'alpha_k') 
#         rho0=self.PL(z, logM200c, 'logrho0')
#         xc=self.PL(z, logM200c, 'xc_k')
#         beta=self.PL(z, logM200c, 'beta_k')

#         pGNFW = lambda p: self.pGNFW(x, 
#             gamma=gamma(p['gamma_k'], p['gamma_k_z'], p['gamma_k_m']), 
#             alpha=alpha(p['alpha_k'], p['alpha_k_z'], p['alpha_k_m']), 
#             rho0=rho0(10**p['logrho0'], p['logrho0_z'], p['logrho0_m']), 
#             xc=xc(p['xc_k'], p['xc_k_z'], p['xc_k_m']), 
#             beta=beta(p['beta_k'], p['beta_k_z'], p['beta_k_m']))
#         return lambda p={}: pc*pGNFW(self.p0 | p)

#     def PGNFW(self, x, P0, xc, gamma, alpha, beta):  # Eq 17
#         return P0 * (x/xc)**gamma * (1+(x/xc)**alpha)**(-beta)
    
#     def P200c(self, z, logM200c, units='cosmo'): 
#         P200c = c.G*(10**logM200c*u.Msun)*200*self.rhoc(z)/(2*self.r200c(z, logM200c))
#         return self.Fb*P200c.to(self.units('pres', units))
    
#     def Pressure1h(self, r, z, logM200c, units='cosmo'):  
#         r, z, logM200c = self.setdim(r, z, logM200c)  # set proper dimensions [nr, nz, nM]
#         P200c = self.P200c(z, logM200c, units)
#         x = r*u.Mpc/self.r200c(z, logM200c)
        
#         gamma=self.PL(z, logM200c, 'gamma_t')
#         alpha=self.PL(z, logM200c, 'alpha_t') 
#         P0=self.PL(z, logM200c, 'P0')
#         xc=self.PL(z, logM200c, 'xc_t')
#         beta=self.PL(z, logM200c, 'beta_t')
        
#         PGNFW = lambda p: self.PGNFW(x, 
#             gamma=gamma(p['gamma_t'], p['gamma_t_z'], p['gamma_t_m']), 
#             alpha=alpha(p['alpha_t'], p['alpha_t_z'], p['alpha_t_m']), 
#             P0=P0(p['P0'], p['P0_z'], p['P0_m']), 
#             xc=xc(p['xc_t'], p['xc_t_z'], p['xc_t_m']), 
#             beta=beta(p['beta_t'], p['beta_t_z'], p['beta_t_m']))
#         return lambda p={}: P200c*PGNFW(self.p0 | p)

#     def prof2h(self, r, z, logM200c): 
#         dndlogm = lambda z, logM200c: self.dndlogm(z, logM200c)
#         bh = lambda z, logM200c: self.bh(z, logM200c)
#         Plin = lambda k, z: self.Plin(k, z)
        
#         V17 = Vikram2017.Profiles(dndlogm=dndlogm, bh=bh, Plin=Plin)
#         logM200c_2h = np.linspace(10, 15, 10)
#         lin2h = V17.twohalo(r, z, logM200c, logM200c_2h)  # linear two-halo calculation
#         return lambda prof, p={}: lin2h(prof(r, z, logM200c_2h)(p))

#     def Density2h(self, r, z, logM200c, units='cosmo'):  # two-halo density component
#         twohalocalc = self.prof2h(r, z, logM200c)
#         return lambda p={}: (self.p0 | p)['A2h_k']*twohalocalc(self.Density1h, p | {par[:-3]: p[par] for par in p if '_2h' in par}).to(self.units('dens', units))

#     def Pressure2h(self, r, z, logM200c, units='cosmo'):  # two-halo pressure component
#         twohalocalc = self.prof2h(r, z, logM200c)
#         return lambda p={}: (self.p0 | p)['A2h_t']*twohalocalc(self.Pressure1h, p ).to(self.units('pres', units))

#     def Pressure(self, r, z, logM200c, units='cosmo'):
#         P1h, P2h = self.Pressure1h(r, z, logM200c, units), self.Pressure2h(r, z, logM200c, units)
#         return lambda p={}: P1h(self.p0 | p) + P2h(self.p0 | p)
    
#     def Density(self, r, z, logM200c, units='cosmo'):
#         p1h, p2h = self.Density1h(r, z, logM200c, units), self.Density2h(r, z, logM200c, units)
#         return lambda p={}: p1h(self.p0 | p) + p2h(self.p0 | p)
    
    
    
# from Models.TargetData import BaseTargetData
# class TargetData(BaseTargetData, Study):  # TODO: In progress
#     path = f"{DATA_PATH}/Results"
#     subs = {
#     'zbin': ['z1', 'z2', 'z3', 'z4'],
#     'deproj' : ['Base', 'cib', 'cib_cibdBeta', 'cib_cibdBeta_cibdT', 'cib_cibdT'],
#     'TCIB': ['10.7', '24.0'],
#     'beta': ['1.0', '1.2', '1.4', '1.6', '1.7', '1.8', '2.0'],
#     }

#     def __init__(self, inputsdict, **inputvars):        
#         self.setup(inputsdict | inputvars)
        
#         self.require(['deproj'])
#         if self.deproj!='cib_cibdBeta_cibdT': # Add values
#             self.subs['TCIB'] = ['10.7']
#         if self.deproj!='cib_cibdBeta': # Add values
#             self.subs['TCIB'] = self.subs['TCIB']+['1.15', '1.30', '1.35', '1.45']

#         if self.deproj=='Base': self.require(['deproj'])
#         else: self.require(['deproj', 'TCIB', 'beta'])

#         ACTDR6 = Coulton2024()
#         self.beam_ells, self.beam_data = ACTDR6.beam_ells, ACTDR6.beam_data


#         # Taking properties from Zhou
#         Zhou = Zhou2023({'zbin':self.zbin,'sample':'main','hemisphere':'combined'})
#         for val in ['zs', 'dNdz', 'dndz', 'dz', 'area', 'logMhMean', 'zMean']:
#             setattr(self, val, getattr(Zhou, val))