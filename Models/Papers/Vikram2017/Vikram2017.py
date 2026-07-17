"""
A Measurement of the Galaxy Group-Thermal Sunyaev-Zel'dovich Effect Cross-Correlation Function

ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
arxiv.org/pdf/1608.04160
"""



    # subs = {}
    # info = {
    #     'ns':1, 'sigma8':0.8, 'Om0':0.27, 'Obl':0.73, 'Ob0':0.044, 'h':0.7,
    #     'MassDef':'200c', 'MassFunc':'Sheth99', 'HaloBias':'Sheth01',
    # }
    
    
    
    # def Fig3(self, width=16, height=10):
    #     return self.plot(filename=['Fig3a','Fig3b', 'Fig3c','Fig3d','Fig3e','Fig3f'], nrow=2, ncol=3, width=width, height=height,
    #         xlabel=r'$r$ [Mpc]', ylabel=r'$\xi^s_{y, g}(r)$',
    #         xlim=(0.01, 10), ylim=[(1e-10, 6.7e-8), (1e-10, 7.7e-8), (1e-10, 1.9e-7), (5e-10, 0.95e-6), (1e-9, 3.6e-6), (1e-9, 1.9e-5)], xscale='log', yscale='log')


class Studies(BaseStudy):  # A Measurement of the Galaxy Group-Thermal Sunyaev-Zel'dovich Effect Cross-Correlation Function, ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
    subs = {}
    info = {
        'ns':1, 'sigma8':0.8, 'Om0':0.27, 'Obl':0.73, 'Ob0':0.044, 'h':0.7,
        'MassDef':'200c', 'MassFunc':'Sheth99', 'HaloBias':'Sheth01',
    }


class Profiles(BaseProfile, Studies.Vikram2017):  # TODO in progress
    models = {}  # only one model
    params = { 
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)
        
        # B11 = Battaglia2011(inputsdict | inputvars, **self.info)
        # self.P1h_del = B11.P_del
        # self.P1h = B11.P
        # self.P200c = B11.P200c

    def twohalo(self, rs, zs, logMs, logMs_2h):  # Eq 8
        self.require(['dndlogm', 'bh', 'Plin'])  # required functions
        
        fft = HaloModels.mcfit_package(rs=rs)  # setup FFT
        ks, FFT3D, IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions
        ks, zs, logMs = np.array(ks, ndmin=1)[:, None, None], np.array(zs, ndmin=1)[:, None], np.array(logMs, ndmin=1)  # Assign proper dimensions [nr, nz, nm]

        prefac = self.bh(zs, logMs)*self.Plin(ks, zs)  # collect factors outside int
        intfac = self.dndlogm(zs, logMs_2h)*self.bh(zs, logMs_2h)  # collect factors inside int: uses M200h instead of other
        P2h = lambda prof1h: prefac*(np.trapz(FFT3D(prof1h)*intfac,logMs_2h*u.dex))[..., None]  # integrate of 2h mass range
        return lambda prof1h: IFFT3D(P2h(prof1h)) *prof1h.unit


class Plots(BasePlots):  # ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
    def Fig3(self, width=16, height=10):
        return self.plot(filename=['Fig3a','Fig3b', 'Fig3c','Fig3d','Fig3e','Fig3f'], nrow=2, ncol=3, width=width, height=height,
            xlabel=r'$r$ [Mpc]', ylabel=r'$\xi^s_{y, g}(r)$',
            xlim=(0.01, 10), ylim=[(1e-10, 6.7e-8), (1e-10, 7.7e-8), (1e-10, 1.9e-7), (5e-10, 0.95e-6), (1e-9, 3.6e-6), (1e-9, 1.9e-5)], xscale='log', yscale='log')
