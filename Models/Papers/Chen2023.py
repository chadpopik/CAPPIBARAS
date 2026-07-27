"""
Thermal Energy Census with the Sunyaev-Zel'dovich Effect of DESI Galaxy Clusters/Groups and Its Implication on the Weak-lensing Power Spectrum

ui.adsabs.harvard.edu/abs/2023ApJ...953..188C
arxiv.org/pdf/2201.12591
"""


from config import *
class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023ApJ...953..188C
    subs = {}
    info = {
        }


class Profiles(BaseProfile, Studies.Chen2023):  # https://arxiv.org/pdf/2201.12591
    models = {}
    params = {
        # fixed
        'mu_e':1.17,
        'mu_p':1.17,
        'gamma':1.17
    }

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars, model=True)

    # def Pdel(self, zs, logMs, units='cosmo'):  # Eq.7
    #     self.require(['rvir'])
    #     _, zs, logMs = self.setdim(1, zs, logMs)  # set proper dimensions [nr, nz, nM]
    #     Tv = 2*c.G*10**logMs*u.Msun*c.m_p*self.p0['mu_e']*(1+zs)/self.rvir(zs, logMs)/c.k_B/3
    #     return Tv*c.k_B

    def conc(self, zs, logMs):
        return 7.85*(10**logMs/2e12)**(-0.081)*(1+zs)**(-0.71)

    # def Pe_del(self, xs, zs, logMs):
    #     xs, zs, logMs = self.setdim(xs, zs, logMs)  # set proper dimensions [nr, nz, nM]
    #     rho_gas = (np.log(1+xs*self.conc(zs, logMs))/(xs*self.conc(zs, logMs)))**(1/(self.p0['gamma']-1))
    #     return rho_gas/c.m_p/self.p0['mu_e'] * u.Msun/u.Mpc**3
    
    def Pe_del(self, xs, zs, logMs):
        xs, zs, logMs = self.setdim(xs, zs, logMs)  # set proper dimensions [nr, nz, nM]
        P_del = (np.log(1+xs*self.conc(zs, logMs))/(xs*self.conc(zs, logMs)))**(1/(self.p0['gamma']-1)+1)
        return P_del
    
    def Pdel(self, zs, logMs, units='cosmo'):  # Eq.7
        self.require(['rhoc', 'rvir'])
        _, zs, logMs = self.setdim(1, zs, logMs)  # set proper dimensions [nr, nz, nM]
        Tv = 2/c.k_B/3 *c.G*10**logMs*u.Msun*c.m_p*self.p0['mu_p']*(1+zs)/self.rvir(zs, logMs)
        return (Tv*c.k_B/c.m_p/self.p0['mu_e'] *1e2*self.rhoc(zs)).to(self.units('pres', units))
    
    
    def Pe1h(self, rs, zs, logMs, units='cosmo'):
        Pdel = self.Pdel(zs, logMs, units)
        Pe_del = self.Pe_del(rs/self.rvir(zs, logMs), zs, logMs)
        return lambda p={}: Pdel*Pe_del
