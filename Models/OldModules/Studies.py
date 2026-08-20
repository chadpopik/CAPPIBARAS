"""
"""


import astropy.units as u
import astropy.constants as c
import numpy as np
import inspect


class BaseStudy:
    # Given a dictionary of input subspecifications and dictionaries of subspecifications with a list of options, check inputs 
    def check_inputs(self, inpdict={}, optdict={}):
        for optkey, optvals in optdict.items():  # for all options that have lists
            if optkey in inpdict and inpdict[optkey] not in optvals:  # if the input contains one BUT it's not in the list
                if isinstance(inpdict[optkey], str) and inpdict[optkey].lower() in [v.lower() for v in optvals]: pass
                else: raise NameError(f"{optkey} {inpdict[optkey]} doesn't exist, choose from: {optvals}")  # print an error
                
    def require_input(self, inpdict, reqlist):  # dict of given inputs, list of required inputs
        for req in reqlist:  # for all required inputs
            if req not in inpdict:  # if input doesn't have it
                raise ValueError(f"Missing {req} is required" )  # print an error

    # Given some dictionary of info and an input, if the info is a dict with a key that matches the input, narrow it down to the value of that key. If it's not, return the info as is
    def spefinfo(self, info, input):
        # If info is a dict with a key that matches the input, narrow it down to the value of that key. If not, ignore it and move on
        try: info = info[input]
        except: pass
        # If info is already a value, return that value with the units
        if not isinstance(info, dict): return info
        # If info is still a dict, rebuild this level of the dict and go down to the next level
        else: return {k: self.spefinfo(v, input) for k, v in info.items()}
        
    # Given some dictionary with a bunch of parameters and the various values they could have, and an input dictionary with keys that should narrow them down, set all those values at atributes of the class after narrowing them down with the input
    def specify_info(self, infodict, inputdict={}):
        # For every key in the info dict that ISN'T the one that sets units
        for infokey in infodict.keys():  
            # set the value of the key to be the info dict (without the units part)
            setattr(self, infokey, {k: v for k, v in infodict[infokey].items() if k != 'units'} if isinstance(infodict[infokey], dict) else infodict[infokey])
            spefval = getattr(self, infokey, None)  # get the value of the key, which is either a dict or a value
            # For every input value, which should be a key that matches to one of the info dict keys
            for inp in inputdict.values(): 
                # Try to use that input key to narrow down every value in the info dict
                # NOTE: if an unrelated input has the same name as a spef, it will ALSO narrow this down. maybe fix this           
                spefval = self.spefinfo(getattr(self, infokey), inp)  # try to use it to narrow it down, also applying units if they exist
                setattr(self, infokey, spefval)  # try to use it to narrow it down

            if isinstance(getattr(self, infokey), dict) or getattr(self, infokey) is None:
                continue # ignore it if it's a dict
                # setattr(self, infokey, None) # set to none if it's still a dict
            elif isinstance(getattr(self, infokey), str):
                continue
            else:
                setattr(self, infokey, spefval*infodict[infokey].get('units', 1) if isinstance(infodict[infokey], dict) else spefval*1)  # try to use it to narrow it down

        return {k: getattr(self, k) for k in infodict.keys()}  # return the info dict with all values narrowed down and set as attributes of the class

    def setup(self, inputs, model=False):
        self.check_inputs(inpdict=inputs, optdict=self.subs)  # check inputs
        infospef = self.specify_info(self.info, inputdict=inputs)  # narrow down info using inputs
        # for infokey, infoval in infospef.items(): setattr(self, infokey, infoval)  # set all info
        for inputkey, inputval in inputs.items(): setattr(self, inputkey, inputval)  # set remaining inputs, overriding info
        self.defineothers()
        self.require = lambda reqlist: self.require_input(inputs, reqlist)

        if model:
            self.check_inputs(inpdict=inputs, optdict=self.models)  # check inputs
            self.p0 = self.specify_info(self.params, inputdict=inputs)  # narrow down info using inputs

    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)



    def defineothers(self):  # define info that can be derived by other params
        for mtype in [f'M{m}{v}' for m in ['h', 's'] for v in ['Mean', 'Max', 'Min']]:
            if getattr(self, mtype, None) is not None and getattr(self, f'log{mtype}', None) is None:
                setattr(self, f'log{mtype}', np.log10(getattr(self, mtype)/u.Msun))
            elif getattr(self, mtype, None) is None and getattr(self, f'log{mtype}', None) is not None:
                setattr(self, mtype, 10**getattr(self, f'log{mtype}')*u.Msun)

        if getattr(self, 'h', None) is not None and getattr(self, 'H0', None) is None: setattr(self, 'H0', getattr(self, 'h')*(100*u.km/u.s/u.Mpc))
        elif getattr(self, 'h', None) is None and getattr(self, 'H0', None) is not None: setattr(self, 'h', getattr(self, 'H0')/(100*u.km/u.s/u.Mpc))

        for O0 in ['Ob0', 'Oc0', 'Om0', 'Ol0']:
            if getattr(self, f'{O0}h2', None) is not None and getattr(self, 'h', None) is not None and getattr(self, O0, None) is None: setattr(self, 'O0', getattr(self, f'{O0}h2')*getattr(self, 'h')**2)
            elif getattr(self, f'{O0}h2', None) is None and getattr(self, 'h', None) is not None and getattr(self, O0, None) is not None: setattr(self, f'{O0}h2', getattr(self, O0)/getattr(self, 'h')**2)

        if getattr(self, 'Om0', None) is not None and getattr(self, 'Ob0', None) is not None and getattr(self, 'Oc0', None) is None: 
            setattr(self, 'Oc0', getattr(self, 'Om0')-getattr(self, 'Ob0'))
        elif getattr(self, 'Om0', None) is None and getattr(self, 'Ob0', None) is not None and getattr(self, 'Oc0', None) is not None: setattr(self, 'Om0', getattr(self, 'Ob0')+getattr(self, 'Oc0'))
        elif getattr(self, 'Om0', None) is not None and getattr(self, 'Ob0', None) is None and getattr(self, 'Oc0', None) is not None: setattr(self, 'Ob0', getattr(self, 'Om0')-getattr(self, 'Oc0'))

        if getattr(self, 'Om0', None) is not None and getattr(self, 'Ob0', None) is not None and getattr(self, 'Fb', None) is None: setattr(self, 'Fb', getattr(self, 'Ob0')/getattr(self, 'Om0'))
        if getattr(self, 'Om0', None) is not None and getattr(self, 'Ob0', None) is None and getattr(self, 'Fb', None) is not None: setattr(self, 'Ob0', getattr(self, 'Fb')*getattr(self, 'Om0'))

        if getattr(self, 'XH', None) is not None and getattr(self, 'Xe', None) is None: setattr(self, 'XH', (2+getattr(self, 'XH'))/(5*getattr(self, 'XH')+3))
        


    # def defineothers(self):  # define info that can be derived by other params
    #     for mtype in [f'M{m}{v}' for m in ['h', 's'] for v in ['Mean', 'Max', 'Min']]:
    #         if mtype in self.info and f'log{mtype}' not in self.info: self.info[f'log{mtype}'] = cycle(self.info[mtype], lambda M: np.log10(M/u.Msun))
    #         elif f'log{mtype}' in self.info and mtype not in self.info: self.info[mtype] = cycle(self.info[f'log{mtype}'], lambda logM: 10**logM*u.Msun)

    #     if 'h' in self.info and 'H0' not in self.info: self.info['H0'] = cycle(self.info['h'], lambda h: h*(100*u.km/u.s/u.Mpc))
    #     elif 'H0' in self.info and 'h' not in self.info: self.info['h'] = cycle(self.info['H0'], lambda h: h/(100*u.km/u.s/u.Mpc))

    #     for O0 in ['Ob0', 'Oc0', 'Om0', 'Ol0']:
    #         if f'{O0}h2' in self.info and 'h' in self.info and O0 not in self.info: self.info[O0] = cycle(self.info[f'{O0}h2'], lambda O0, h=self.info['h']: O0/h**2)
    #         elif O0 in self.info and 'h' in self.info and f'{O0}h2' not in self.info: self.info[f'{O0}h2'] = cycle(self.info[O0], lambda O0, h=self.info['h']: O0/h**2)

    #     if 'Om0' in self.info and 'Ob0' in self.info and 'Oc0' not in self.info: self.info['Oc0'] = self.info['Om0']-self.info['Ob0']
    #     elif 'Oc0' in self.info and 'Ob0' in self.info and 'Om0' not in self.info: self.info['Om0'] = self.info['Oc0']+self.info['Ob0']

    #     if 'Om0' in self.info and 'Ob0' in self.info and 'Fb' not in self.info: self.info['Fb'] = self.info['Ob0']/self.info['Om0']
    #     elif 'Om0' in self.info and 'Fb' in self.info and 'Ob0' not in self.info: self.info['Ob0'] = self.info['Fb']*self.info['Om0']

    #     if 'XH' in self.info and 'Xe' not in self.info:  self.info['Xe'] = (2+self.info['XH'])/(5*self.info['XH']+3)
    
# def cycle(d, f):
#     argnames = list(inspect.signature(f).parameters)
#     try: return f(*[d[p] for p in argnames])  # try to apply the function normally
#     except:  # if it failed, one value is still a dict
#         def all_keys(d):
#             return [k for k in d.keys()] + [kk for v in d.values() if isinstance(v, dict) for kk in all_keys(v)]
#         unique_keys = list(dict.fromkeys(k for p in argnames for k in (all_keys(d[p]) if isinstance(d[p], dict) else [])))
        
#         outdict={}
#         for v in unique_keys:
#             if spefinfo(d, v)==d: continue # if nothing was specified, skip it
#             elif v not in all_keys(outdict): # if something was specific, make a new key for it
#                 outdict[v] = cycle(spefinfo(d, v), f)
#         return outdict
    
def cycle(d, f):
    return {k: cycle(v, f) if isinstance(v, dict) else f(v) for k, v in d.items()} if isinstance(d, dict) else f(d)
            
            
            
            
    
    
class Zheng2005(BaseStudy):  # Theoretical Models of the Halo Occupation Distribution: Separating Central and Satellite Galaxies, ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
    subs = {}
    info = {'MassDef': 'vir',
        }

class Zehavi2005(BaseStudy):  # The Luminosity and Color Dependence of the Galaxy Correlation Function, ui.adsabs.harvard.edu/abs/2005ApJ...630....1Z
    subs = {}
    info = {
        }
    
class Nagai2007(BaseStudy):  # Effects of Galaxy Formation on Thermodynamics of the Intracluster Medium, ui.adsabs.harvard.edu/abs/2007ApJ...668....1N
    subs = {}
    info = {
        # Cosmological Parameters, 2p1/2p3/3p2
        'Om0':0.3, 'Ob0':0.04286, 'h':0.7, 'sigma8':0.9,
        'Ol0':0.7, 'Fb':0.175,
        'MassDef':'500c', 'Concentration': 'Constant', # Mass definition, S2p3/Eq11
        # right after eq 1/2/3
        'mu':0.59, 'mu_e':'1.14'
    }


class Arnaud2010(BaseStudy):  # The universal galaxy cluster pressure profile from a representative sample of nearby systems (REXCESS) and the YSZ - M500 relation, ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
    subs = {}
    info = {
        # Cosmological Parameters, 1p-1
        'h':0.7, 'Om0':0.3, 'Ol0':0.7, 'Concentration': 'Constant', 'MassDef':'500c',
    }


class Battaglia2011(BaseStudy):  # On the Cluster Physics of Sunyaev-Zel'dovich and X-Ray Surveys. II. Deconstructing the Thermal SZ Power Spectrum, ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
    subs = {}
    info = {
        # Cosmological Parameters, 2p1/2p3/3p2
        'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'ns':0.96, 'sigma8':0.8, 'h':0.7, 'XH':0.76,
        'mdef':'200c',  # Mass definition, S2p3/Eq11
    }

class Ahn2013(BaseStudy):  # The Tenth Data Release of the Sloan Digital Sky Survey: First Spectroscopic Data from the SDSS-III Apache Point Observatory Galactic Evolution Experiment, ui.adsabs.harvard.edu/abs/2014ApJS..211...17A
    subs = {'DR': ['DR10', 'DR12']
    }
    info = {
        'area': {'DR12': 9376, 'DR10': 6373.2},
        }

    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)
    
class Alam2015(BaseStudy):  # The Eleventh and Twelfth Data Releases of the Sloan Digital Sky Survey: Final Data from SDSS-III, ui.adsabs.harvard.edu/abs/2015ApJS..219...12A
    subs = {'DR': ['DR10', 'DR12']
    }
    info = {
        'area': {'DR12': 9376, 'DR10': 6373.2},
        }

    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)

class Planck2013(BaseStudy):  # Planck intermediate results. V. Pressure profiles of galaxy clusters from the Sunyaev-Zeldovich effect, ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subs = {}

    info = {'Om0':0.3, 'Ol0':0.7,'h':0.7,'MassDef':'500c',
        }
    
class More2015(BaseStudy):  # The Weak Lensing Signal and the Clustering of BOSS Galaxies. II. Astrophysical and Cosmological Constraints, ui.adsabs.harvard.edu/abs/2015ApJ...806....2M
    subs = {'mbin': ['MA', 'MB', 'MC']}
    info = {
        # Free cosmological parameters, Table 1
        "Om0": {"MA": 0.310, "MB": 0.306, "MC": 0.304},
        "sigma8": {"MA": 0.785, "MB": 0.839, "MC": 0.813},
        "100*Ob0h2": {"MA": 2.228, "MB": 2.226, "MC": 2.222},
        "ns": {"MA": 0.964, "MB": 0.963, "MC": 0.961},
        "h": {"MA": 0.703, "MB": 0.700, "MC": 0.695},
        # Sample info, Section 2p2
        'logMsMin': {"MA": 11.10, "MB": 11.30, "MC": 11.40},
        'logMsMax': {"MA": 12.00, "MB": 12.0, "MC": 12.0},
        'Ngal': {"MA": 400916, "MB": 196578, "MC": 116682},
        'ngal': {"MA": 3e-4, "MB": 1.5e-4, "MC": 0.8e-4},  # (Mpc/h)^{-3}
        # Model definitions, Section 3.1pLast
        'mdef': '200m',  # M200b, 200 times overdense wrt background matter density
        'HMFModel':'Tinker08' ,'BiasModel': 'Tinker10', 'ConcModel':'Maccio08',
        
        # Free model parameters, Table 1
        "M_stellar_11": {"MA": 0, "MB": 0, "MC": 0},  # describes the average stellar mass of galaxies, [10^11 h^(-2) Msun]
        "R_c": {"MA": 0.98, "MB": 1.01, "MC": 1.02},  # normalization of the concentration mass relation with respect to the one obtained from simulations
        "psi": {"MA": 0.93, "MB": 0.93, "MC": 0.94},  # nuisance parameters
    }

    # info['MhM_stellar_11Min'] = cycle(info['M_stellar_11'], lambda M, h=info['h']: M*u.Msun/h)  # TODO: how to handle multiple h values??
    info['Ob0h2'] = cycle(info['100*Ob0h2'], lambda o: o/100)


class Battaglia2016(BaseStudy):  # The tau of galaxy clusters, ui.adsabs.harvard.edu/abs/2016JCAP...08..058B
    subs={}
    info = {
        # sim's cosmo params, B15.3.P2
        'XH':0.76, 'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'h':0.72, 'ns':0.96, 'sigma8':0.8,
        'MassDef':'200c',  # Mass definition, B15.T2
        'MassFunc': 'Tinker08',
    }




class Vikram2017(BaseStudy):  # A Measurement of the Galaxy Group-Thermal Sunyaev-Zel'dovich Effect Cross-Correlation Function, ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
    subs = {}
    info = {
        'ns':1, 'sigma8':0.8, 'Om0':0.27, 'Obl':0.73, 'Ob0':0.044, 'h':0.7,
        'MassDef':'200c', 'MassFunc':'Sheth99', 'HaloBias':'Sheth01',
    }


class Kravtsov2018(BaseStudy):  # Stellar Mass—Halo Mass Relation and Star Formation Efficiency in High-Mass Halos, ui.adsabs.harvard.edu/abs/2018AstL...44....8K
    subs = {}
    info = {
        # fixed cosmo params, Section 1pLast
        'Om0':0.27, 'Ob0':0.0469, 'h':0.7, 'sigma8': 0.82, 'ns':0.95,
    }

class Planck2018(BaseStudy):  # Planck 2018 results. VI. Cosmological parameters, ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P
    subs = {}
    info = { # Table 2 TT,TE,EE+lowE+lensing
        'H0': 67.36*u.km/u.s/u.Mpc, 'Ob0h2':0.02237, 'Oc0h2':0.1200,'ns':0.9649, 'sigma8':0.8120, 'Om0':0.3166, 'Ol0':0.6834
        }


class Koukoufilippas2020(BaseStudy):  # arxiv.org/abs/1909.09102
    subs={'sample':['2MPZ','WIxSC-1','WIxSC-2','WIxSC-3','WIxSC-4','WIxSC-5'],}
    info={}


class Naess2020(BaseStudy):  # ui.adsabs.harvard.edu/abs/2020JCAP...12..046N
    subs={}
    info={}

class To2020(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021PhRvL.126n1301T
    subs = {}
    info = {
        }
    
class Schaan2021(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021PhRvD.103f3513S
    subs = {
        'sample' : ['cmass', 'lowz'],  # galaxy sample (CMASS M from DR12 not available for everything)
    }
    info = {
        'T_CMB':2.726,  # CMB temp [K], Section F.1p8
        'v_rms': {'lowz':320, 'cmass':313},  # rms velocity [km/s] at mean redshifts, Section F.1p8/F.2p1
        'area': 6000,  # area of overlap between ACT and BOSS [deg^2], TODO 1: assumed
        'mdef':'vir', 'MhMean': {'lowz':5e13, 'cmass':3e13},  # halo mass definition and mean halo masses, Figure 3
        'MsMax': 5.5e11, 'MhMax': 1e14, # max stellar mass and halo mass, Section IV.Ep2
        'zMin':0.4, 'zMax':0.7,  # redshift range, Section IIp1
        'zMean': {'lowz':0.31, 'cmass':0.55},  # mean redshift, Figure 2 (says 0.55 everywhere else in the paper)
        'Ngal_catalog':{'lowz':218905, 'cmass': 501844, 'CMASSm':777202},  # total galaxies in BOSS catalog, Section III.Ap2
        'Ngal_overlap': {'lowz':151713, 'cmass': 325518, 'CMASSm':385137},  # galaxies in ACT BOSS overlap, Section III.Ap2
        'Ngal_masked': {'lowz':145714, 'cmass': 312708, 'CMASSm':368701},  # galaxies in overlap after masking, Section III.Ap2
        'Ngal': {'lowz':134702, 'cmass':311309, 'CMASSm':360084},  # final galaxy count after applying upper mass limit, Section III.Ap2
        }

    info['T_CMB'] = cycle(info['T_CMB'], lambda T: T *u.K)
    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)
    info['MhMean'] = cycle(info['MhMean'], lambda M: M *u.Msun)
    info['MsMax'] = cycle(info['MsMax'], lambda M: M *u.Msun)
    info['MhMax'] = cycle(info['MhMax'], lambda M: M *u.Msun)


class Amodeo2021(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021PhRvD.103f3514A
    subs = {}
    info = {
        # cosmological parameters, Ip10, IIA.Ap4/p5
        'Om0': 0.25, 'Ob0': 0.044, 'OL0': 0.75, 'h': 0.7,  
        'v_rms': 1.06e-3, 'XH':0.76,   # RMS of peculiar velocites [v/c] and hydrogen mass fraction
        'T_CMB': 2.725*u.K,  # mop-c-gt, mopc.py
        # Model implementation, Ip10, Appendix A p2
        'MassDef': 'fof', 'MassFunc':'sheth99', 'HaloBias':'sheth01',
    }

class Moser2021(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
    subs = {}
    info = Amodeo2021.info
    
    
class Linke2022(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
    subs = {'sample':['MS', 'KVG'],
            'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    info = {
        # best fit params, unclear what for:
        'A': {'MS': {'r': 5.31, 'b': 5.31}, 'KVG': {'r': 1.62, 'b': 1.62}},
        'epsilon': {'MS': {'r': 0.69, 'b': 0.69}, 'KVG': {'r': 0.99, 'b': 0.99}},

        # Cosmo params 1.p8
        'Om0':{'MS': 0.25, 'KVG': 0.315}, 'Ob0':{'MS': 0.045, 'KVG': 0.049}, 'H0':{'MS': 73, 'KVG': 67.4}, 'sigma8':{'MS': 0.9, 'KVG': 0.811},
        'mdef': '200m', 'zMax': 0.5,
        # Msun/h^2, these cuts are just for sims
        'MhMin':{'MS': 10e11}, 'MhMax': {'MS':10e15},
    }
    info['H0'] = cycle(info['H0'], lambda H: H *u.km/u.s/u.Mpc)
    info['MhMin'] = cycle(info['MhMin'], lambda M: M *u.Msun)
    info['MhMax'] = cycle(info['MhMax'], lambda M: M *u.Msun)



class White2022(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022JCAP...02..007W
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'],} # photometric redshift subsmaple
    info = {
        'SN': {'z1':4.02, 'z2':2.24, 'z3':2.07, 'z4':2.26},  # shot noise level [1e6]
        'zeff': {'z1':0.47, 'z2':0.62, 'z3':0.78, 'z4':0.91},  # effective z at which power spec is calculate approx
        'lmax': {'z1':250, 'z2':300, 'z3':350, 'z4':400},  # max ell used in fits
        'lSN': {'z1':400, 'z2':530, 'z3':575, 'z4':425},  # ell where SN equals modeled auto-spec power
    }




class Kusiak2022(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022PhRvD.106l3517K
    subs = {'sample':['Blue', 'Green', 'Red'],}

    info = {
        # best fit HOD params
        "ASNe7": {"Blue": -0.16, "Green": 1.35, "Red": 27.95},
        # Table II
        'zMean': {'Blue': 0.6, 'Green': 1.1, 'Red': 1.5},
        'ndens': {'Blue': 3409, 'Green': 1846, 'Red': 144},
        'area': 0.586*4 * np.pi * (180/np.pi)**2*u.deg**2,
        
        # fixed cosmo params
        'Oc0h2': 0.11933, 'Ob0h2': 0.02242, 'h':0.6766, 'ns':0.9665, 'lnAsn10': 3.047, 'kpivot':0.05, 'tau_reio':0.0561,  # Ip7
        # HaloModel choices, Eq 10&30, Section IpLast
        'MassDef': '200c', 'Concentration': 'Bhattacharya13', 'MassFunc': 'Tinker08', 'HaloBias': 'Tinker10',
        # Other info
        'MhMin': 7e8, 'MhMax': 3.5e15,  # Msun/h
        'zMin_hmod': 0.005, 'zMax_hmod': 4,
        'zMin': 0, 'zMax': 2,
        'logM0': 0,
            }

    info['MhMin'] = cycle(info['MhMin'], lambda M, h=info['h']: M*u.Msun/h)
    info['MhMax'] = cycle(info['MhMax'], lambda M, h=info['h']: M*u.Msun/h)
    info['ndens'] = cycle(info['ndens'], lambda n: n/u.deg**2)


class Zhou2023(BaseStudy):  # arxiv.org/abs/2309.06443
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }
    info = {
        'area': {  # area of survey [deg^2]], 2.1p2/3.2p3
                'main':{'combined':16700, 'north':4200, 'south':12500},
                'extended':{'combined':230, 'north':100, 'south':130}},
        'logMhMean': {'z1': 13.40, 'z2': 13.40, 'z3': 13.24, 'z4': 13.24},  # mean halo mass taken from Yuan 2023 [Msun?], 6.p2
        # mean number density, mean redshift, min/max photometrix redshift bounds,  T1/T2/T3
        'nGal': {  # [deg^-2]
            'main': {'all':600, 'z1': 81.9, 'z2': 148.1, 'z3': 162.4, 'z4': 148.3},
            'ext': {'all':1669, 'z1': 185.5, 'z2': 311.0, 'z3': 422.6, 'z4': 438.4},},
        'zMean': {
            'main': {'z1': 0.470, 'z2': 0.628, 'z3': 0.791, 'z4': 0.924},
            'ext': {'z1': 0.467, 'z2': 0.633, 'z3': 0.794, 'z4': 0.929},},
        'zpMin': {
            'main': {
                'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, 'z3': 0.719, 'z4': 0.851},
                'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, 'z3': 0.713, 'z4': 0.860},},
            'ext': {
                'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, '3': 0.719, 'z4': 0.854},
                'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, '3': 0.713, 'z4': 0.860},},},
        'zpMax': {
            'main': {
                'north': {'all':1.024, 'z1': 0.545, 'z2': 0.719, 'z3': 0.851, 'z4': 1.024},
                'south': {'all':1.020, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.020},},
            'ext': {
                'north': {'all':1.010, 'z1': 0.545, 'z2': 0.719, 'z3': 0.854, 'z4': 1.010},
                'south': {'all':1.000, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.000},},},
    }
    for val in ['zpMin', 'zpMax']:  # Assumeding combined uses south limits
        for samp in info[val].keys(): 
            info[val][samp]['combined'] = info[val][samp]['south']
    info['area'] = cycle(info['area'], lambda a: a*u.deg**2)
    info['nGal'] = cycle(info['nGal'], lambda n: n /u.deg**2)
    
    
class Chen2023(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023ApJ...953..188C
    subs = {}
    info = {
        }


class Gao2023(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
    subs = {}
    info = {        
        # fixed cosmo info
        'h':0.71, 'Om0':0.268, 'Ol0':0.732,
        'mdef':'vir',  # Current Virial Mass
        'area': 140,  # covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
    }
    info['area'] = cycle(info['area'], lambda a: a*u.deg**2)





class Yuan2023(BaseStudy):  # ui.adsabs.harvard.edu/abs/2024MNRAS.530..947Y
    subs = {'zbin': ['LRG1', 'LRG2', 'QSO', 'LRG3', 'LRG4']}
    info = {
        "f_sat": {"LRG1": 0.089, "LRG2": 0.104, "QSO": 0.05, "LRG3": 0.110, "LRG4": 0.151},
        "logMhMean": {"LRG1": 13.42, "LRG2": 13.26, "QSO": 12.74, "LRG3": 13.29, "LRG4": 13.00},  # Msun/h
        "b_lin": {"LRG1": 1.94, "LRG2": 2.11, "QSO": 2.56, "LRG3": 2.31, "LRG4": 2.13},
        
        # fixed cosmo params, 1.p7
        'Oc0h2': 0.1200, 'Ob0h2': 0.02237, 'sigma8': 0.811355, 'ns': 0.9649, 'h': 0.6736, 'w0':-1, 'wa':0,
        'mdef': '200c',  # M not clear, maybe same as zheng 2005/2007? or cmass?
        'MhMin': 1.3e11,  # Msun/h
        'zMin': {'LRG1': 0.6, 'LRG2': 0.8, 'QSO': 2.1, 'LRG3': 0.95, 'LRG4': 0.8},
        'zMax': {'LRG1': 0.4, 'LRG2': 0.6, 'QSO': 0.8, 'LRG3': 0.8, 'LRG4': 0.95},
    }
    info['MhMin'] = cycle(info['MhMin'], lambda M, h=info['h']: M*u.Msun/h)
    info['logMhMean'] = cycle(info['logMhMean'], lambda p, h=info['h']: np.log10(10**p/h))



class Xu2023(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023ApJ...944..200X
    subs = {'sample': ['Main', 'LOWZ', 'CMASS'],
            'form': ['BP13', 'DP']}
    info = {
        # Fixed cosmo parameters, Section 3.1
        'Om0':0.268, 'Ol0':0.732, 'sigma8':0.831, 'h':0.71,
        # Mass definition, Eq 8 
        'mdef': 'vir',  # virial mass of the halo at the time when the galaxy was last the central dominant object
    }



class Kou2023(BaseStudy):  # ui.adsabs.harvard.edu/abs/2023A%26A...675A.149K
    subs = {'mbin':['M1', "M2", "M3", "M4"],}
    info = {
        # best fit HOD parameters
        "A": {"M1": 0.981, "M2": 0.965, "M3": 0.956, "M4": 0.961},  # cross-correlation amplitude
        "beta_m": {"M1": 4.97, "M2": 5.91, "M3": 4.16, "M4": 10},  # matter density profile
        
        # fixed cosmological parameters
        'h':0.6766, 'Ob0h2':0.02242, 'Oc0h2':0.11933, 'tau':0.0561, 'ns':0.9665, 'sigma8':0.8102, # 5.1p3
        'mdef': '200m',  # region in which the average density is ∆ = 200 times the cosmic mean density
        'MassFunc': 'Tinker08', 'Concentration':'Dolag04',
        'zlims': [0.47, 0.59],  # redshift range of selected galaxies
        'zmed': 0.53,  # median redshift
        'logMsMin': {'M1': 10.8, 'M2': 11.1, 'M3': 11.25, 'M4': 11.4},  # minimum stellar mass of selected
        'c0': 9.59, 'alpha_c': -0.102,  # concentration parameters, Eq47
    }
    
    def conc(self, z, logM):  # Eq 47
        return self.c0/(1+z) * (10**logM/(10**14))**self.alpha_c


class Coulton2024(BaseStudy):  # ui.adsabs.harvard.edu/abs/2024PhRvD.109f3530C
    subs={}
    info={'area': 12200
          }
    info['area'] = cycle(info['area'], lambda a: a*u.deg**2)

class Hadzhiyska2025A(BaseStudy):  # Missing baryons recovered: a measurement of the gas fraction in galaxies and groups with thekinematic Sunyaev-Zel’dovich effect and CMB lensing, ui.adsabs.harvard.edu/abs/2025PhRvD.112l3507H
    subs = {
    }

    info = {'h': 0.6736, 'MassDef': 'vir'
    }
    #Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0
    
class Hadzhiyska2025B(BaseStudy):  # Probing cosmic velocities with the pairwise kinematic Sunyaev-Zel'dovich signal in DESI Bright Galaxy Sample DR1 and ACT DR6, ui.adsabs.harvard.edu/abs/2025arXiv251014135H
    subs = {'mbin': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
    }

    info = Planck2018.info | { # says it uses Planck2018 cosmology
        "MassDef": 'vir',
        "logMsMin": {"M1": 11.861, "M2": 11.918, "M3": 11.831, "M4": 11.750, "M5": 11.947, "M6": 12.183},
        "fsat": {"M1": 0.123, "M2": 0.147, "M3": 0.206, "M4": 0.295, "M5": 0.333, "M6": 0.343},
        "logMhMean": {"M1": 13.135, "M2": 13.184, "M3": 13.266, "M4": 13.310, "M5": 13.370, "M6": 13.456},
        "blin": {"M1": 1.155, "M2": 1.190,  "M3": 1.258, "M4": 1.314, "M5": 1.384, "M6": 1.475},
        
    }

   
class RiedGuachalla2025(BaseStudy):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112j3512R
    subs = {}  # subset of galaxy selection
    info = {
        }



class Hadzhiyska2025(BaseStudy):  # arxiv.org/abs/2407.07152
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],}
    info={}
    
#     # info = {'ngal': {'ext_DR9_z1': 963631, 'ext_DR9_z2': 1658313, 'ext_DR10_z3': 1951646, 'ext_DR10_z4':1690171, 'ext_all':6850072},
#     # # TODO: come back to this
#     # }
#     info = {'name':'Hadzhiyska 2025'}




class Liu2025(BaseStudy):  # Measurements of the thermal Sunyaev-Zel'dovich effect with ACT and DESI luminous red galaxies, ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subs = {
    }
    info = {
    }


class Hadzhiyska2026(BaseStudy):  # ui.adsabs.harvard.edu/abs/2025arXiv251014135H
    subs={}
    info = {
        'area': 12200,  # NOTE: area value not given, assumed
        'h': 0.7,  # NOTE: h value not paper, assumed
    }
    info['area'] = info['area']*u.deg**2


class Moore2026(BaseStudy):
    subs={}
    info = {'area': 16700*u.deg**2,  # assuming the same as XCorr LRGs
            }
    
class Popik2026(BaseStudy):  # In progress
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],
    }
    info = {
        'name':'Popik 2025'
        }

