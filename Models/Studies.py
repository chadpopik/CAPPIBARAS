"""
Required/relevant information from studies used in this CAPPIBARAS, either for data, measurements, or models. Informaiton includes cosmological parameters, properties of data, models used, etc.
"""


import astropy.units as u
import astropy.constants as c
import numpy as np
import inspect

import Models.HaloModels as HaloModels


class BaseStudy(HaloModels.BaseModel):
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def setup(self, inputs, model=False):
        self.check_inputs(inpdict=inputs, optdict=self.subs)  # check inputs
        infospef = self.specify_info(self.info, inputdict=inputs)  # narrow down info using inputs
        for infokey, infoval in infospef.items(): setattr(self, infokey, infoval)  # set all info
        for inputkey, inputval in inputs.items(): setattr(self, inputkey, inputval)  # set remaining inputs, overriding info
        self.defineothers()
        self.require = lambda reqlist: self.require_input(inputs, reqlist)

        if model:
            self.check_inputs(inpdict=inputs, optdict=self.models)  # check inputs
            self.p0 = self.specify_info(self.params, inputdict=inputs)  # narrow down info using inputs

    def specify_info(self, infodict, inputdict={}):
        spefinfodict = infodict.copy()  # make a new dict
        for infokey in spefinfodict.keys():  # for all the general info
            for inp in inputdict.values(): # for every input 
                # NOTE: if an unrelated input has the same name as a spef, it will ALSO narrow this down. maybe fix this
                spefinfodict[infokey]=spefinfo(spefinfodict[infokey], inp)  # try to use it to narrow it down
        return {k:v for k, v in spefinfodict.items() if not isinstance(v, dict)}
    
    
    
    
    def defineothers(self):  # define info that can be derived by other params
        for mtype in [f'M{m}{v}' for m in ['h', 's'] for v in ['Mean', 'Max', 'Min']]:
            if hasattr(self, mtype) and not hasattr(self, f'log{mtype}'): setattr(self, f'log{mtype}', np.log10(getattr(self, mtype)/u.Msun))
            elif not hasattr(self, mtype) and hasattr(self, f'log{mtype}'):
                setattr(self, mtype, 10**getattr(self, f'log{mtype}')*u.Msun)

        if hasattr(self, 'h') and not hasattr(self, 'H0'): setattr(self, 'H0', getattr(self, 'h')*(100*u.km/u.s/u.Mpc))
        elif not hasattr(self, 'h') and hasattr(self, 'H0'): setattr(self, 'h', getattr(self, 'H0')/(100*u.km/u.s/u.Mpc))

        for O0 in ['Ob0', 'Oc0', 'Om0', 'Ol0']:
            if hasattr(self, f'{O0}h2') and hasattr(self, 'h') and not hasattr(self, O0): setattr(self, 'O0', getattr(self, f'{O0}h2')*getattr(self, 'h')**2)
            elif not hasattr(self, f'{O0}h2') and hasattr(self, 'h') and hasattr(self, O0): setattr(self, f'{O0}h2', getattr(self, O0)/getattr(self, 'h')**2)

        if hasattr(self, 'Om0') and hasattr(self, 'Ob0') and not hasattr(self, 'Oc0'): setattr(self, 'Oc0', getattr(self, 'Om0')-getattr(self, 'Ob0'))
        elif not hasattr(self, 'Om0') and hasattr(self, 'Ob0') and hasattr(self, 'Oc0'): setattr(self, 'Om0', getattr(self, 'Ob0')+getattr(self, 'Oc0'))
        elif hasattr(self, 'Om0') and not hasattr(self, 'Ob0') and hasattr(self, 'Oc0'): setattr(self, 'Ob0', getattr(self, 'Om0')-getattr(self, 'Oc0'))

        if hasattr(self, 'Om0') and hasattr(self, 'Ob0') and not hasattr(self, 'Fb'): setattr(self, 'Fb', getattr(self, 'Ob0')/getattr(self, 'Om0'))
        if hasattr(self, 'Om0') and not hasattr(self, 'Ob0') and hasattr(self, 'Fb'): setattr(self, 'Ob0', getattr(self, 'Fb')*getattr(self, 'Om0'))
        
        if hasattr(self, 'XH') and not hasattr(self, 'Xe'): setattr(self, 'XH', (2+getattr(self, 'XH'))/(5*getattr(self, 'XH')+3))
        
def spefinfo(info, input):
    try: info = info[input]  # try to narrow down the current info input
    except: pass  # leave as is otherwise
    if not isinstance(info, dict): return info  # if it's not a dict, return the value
    return {k: spefinfo(v, input) for k, v in info.items()}  # otherwise recursively rebuild the dict


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
            
            

class Hadzhiyska2025B(BaseStudy):  # In progress
    subs = {'mbin': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
    }

    info = {
        "logMsMin": {"M1": 11.861, "M2": 11.918, "M3": 11.831, "M4": 11.750, "M5": 11.947, "M6": 12.183},
        "fsat": {"M1": 0.123, "M2": 0.147, "M3": 0.206, "M4": 0.295, "M5": 0.333, "M6": 0.343},
        "logMhMean": {"M1": 13.135, "M2": 13.184, "M3": 13.266, "M4": 13.310, "M5": 13.370, "M6": 13.456},
        "blin": {"M1": 1.155, "M2": 1.190,  "M3": 1.258, "M4": 1.314, "M5": 1.384, "M6": 1.475},
    }
   
        
        
class Jenna_Catalog(BaseStudy):
    subs={}
    info = {'area': 16700,  # assuming the same as XCorr LRGs
            }
    info['area'] = info['area']*u.deg**2

                
                
class Popik2025(BaseStudy):  # In progress
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],
    }

    info = {
        'name':'Popik 2025'
        }



class RiedGuachalla2025(BaseStudy):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112j3512R
    subs = {'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],}  # subset of galaxy selection
    info = {
        'area': 4300,  # overlapping region of ACT and DESI [deg^2], F1 and III.B.p4
        'logMhMean':13.4,  # estimated mean halo mass of LRG [Msun/h], III.B.p5
        # spectroscopic redshift bins, III.B.p6
        'zMin': {'all':0.4, 'z_1':0.4, 'z_2':0.6, 'z_3':0.8, 'z_4':0.9},
        'zMax': {'all':1.1, 'z_1':0.6, 'z_2':0.8, 'z_3':0.95, 'z_4':1.1},
        # stellar mass bins [Msun], III.B.p7
        'logMsMin': {'all':10.5, 'mass_1':10.5, 'mass_2':11.2, 'mass_3':11.4, 'mass_4':11.6},
        'logMsMax': {'all':12.5, 'mass_1':11.2, 'mass_2':11.4, 'mass_3':11.6, 'mass_4':12.5},
        # subsample info, T2
        'zMean': {'all':0.74, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.76, 'mass_2':0.75, 'mass_3':0.71, 'mass_4':0.69},  # mean redshift
        'zMed': {'all':0.75, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.79, 'mass_2':0.76, 'mass_3':0.70, 'mass_4':0.67},  # median redshift
        'MsMean': {'all':2.2, 'z_1':2.4, 'z_2':2.3, 'z_3':2.0, 'z_4':2.1, 'mass_1':1.2, 'mass_2':2.0, 'mass_3':3.0, 'mass_4':5.1},  # mean stellar mass [10e11 Mstar/Msun]
        'NGal': {'all':825283, 'z_1':195877, 'z_2':235620, 'z_3':235620, 'z_4':96346, 'mass_1':244932, 'mass_2':320914, 'mass_3':194037, 'mass_4':53997},  # number of galaxies
        }
    # Assume z/m bins have same m/z limits as all
    for b in [1, 2, 3, 4]:
        info['zMin'][f'mass_{b}'] = info['zMin']['all']
        info['zMax'][f'mass_{b}'] = info['zMax']['all']
        info['logMsMin'][f'z_{b}'] = info['logMsMin']['all']
        info['logMsMax'][f'z_{b}'] = info['logMsMax']['all']
        
    info['area'] = cycle(info['area'], lambda a: a*u.deg**2)
    info['MsMean'] = cycle(info['MsMean'], lambda M: M*1e11*u.Msun)


class Hadzhiyska2025(BaseStudy):  # arxiv.org/abs/2407.07152
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],}
    info={}
    
#     # info = {'ngal': {'ext_DR9_z1': 963631, 'ext_DR9_z2': 1658313, 'ext_DR10_z3': 1951646, 'ext_DR10_z4':1690171, 'ext_all':6850072},
#     # # TODO: come back to this
#     # }
#     info = {'name':'Hadzhiyska 2025'}




class Liu2025(BaseStudy):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
    }
    info = {
        'area': 7326,  # [deg^2] area of ACT DESI overlap, Fig 1
        'logMhMean': {'z1':13.40, 'z2':13.40, 'z3':13.24, 'z4':13.24},  # rough mean halo mass taken from Yuan 2023, III.A.p5
        # mean redshift, mean number density, objects in catalog, objects in ACT/DESI overlap, T1
        'zMean': {'z1':0.470, 'z2':0.628, '3':0.791, 'z4':0.924},
        'nGal': {'z1':81.9, 'z2':148.1, 'z3':162.4, 'z4':148.3},  # [deg^-2]
        'NGal_unmasked': {'z1':1118496, 'z2':2031303, 'z3':2240982, 'z4':2049158},
        'NGal': {'z1':332280, 'z2':608100, 'z3':671738, 'z4':615543},
    }
    info['area'] = cycle(info['area'], lambda a: a*u.deg**2)
    info['nGal'] = cycle(info['nGal'], lambda n: n /u.deg**2)



class Coulton2024(BaseStudy):  # ui.adsabs.harvard.edu/abs/2024PhRvD.109f3530C
    subs={}
    info={}


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



class Linke2022(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
    subs = {'sample':['MS', 'KVG'],
            'color':['r', 'b']}  # TODO: There are actually many further subsamples cut by stellar mass
    info = {
        # best fit params, unclear what for:
        'f_a': {'MS': {'r': 1.49, 'b': 0.88}, 'KVG': {'r': 1.27, 'b': 0.83}},
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
        'name':'White 2022',
        'area': 18000,  # area of DESI survey, [deg^2], Figure 1
        # info, Table 1
        'zMean': {'z1':0.47, 'z2':0.63, 'z3':0.79, 'z4':0.92},  # mean redshift
        'ndens': {'z1':83, 'z2':149, 'z3':162, 'z4':149},  # galaxy number desnity
        'SN': {'z1':4.02, 'z2':2.24, 'z3':2.07, 'z4':2.26},  # shot noise level [1e6]
        'zeff': {'z1':0.47, 'z2':0.62, 'z3':0.78, 'z4':0.91},  # effective z at which power spec is calculate approx
        'lmax': {'z1':250, 'z2':300, 'z3':350, 'z4':400},  # max ell used in fits
        'lSN': {'z1':400, 'z2':530, 'z3':575, 'z4':425},  # ell where SN equals modeled auto-spec power
    }

    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)
    info['ndens'] = cycle(info['ndens'], lambda n: n /u.deg**2)



class Kusiak2022(BaseStudy):  # ui.adsabs.harvard.edu/abs/2022PhRvD.106l3517K
    subs = {'sample':['Blue', 'Green', 'Red'],}

    info = {
        # best fit HOD params
        "ASNe7": {"Blue": -0.16, "Green": 1.35, "Red": 27.95},
        
        # fixed cosmo params
        'Oc0h2': 0.11933, 'Ob0h2': 0.02242, 'h':0.6766, 'ns':0.9665, 'lnAsn10': 3.047, 'kpivot':0.05,'tau_reio':0.0561,  # Ip7
        # HaloModel choices, Eq 10&30, Section IpLast
        'MassDef': '200c', 'Concentration': 'Bhattacharya13', 'MassFunc': 'Tinker08', 'HaloBias': 'Tinker10',
        # Table II
        'zMean': {'Blue': 0.6, 'Green': 1.1, 'Red': 1.5},
        'ndens': {'Blue': 3409, 'Green': 1846, 'Red': 144},
        
        # Other info
        'MhMin': 7e8, 'MhMax': 3.5e15,  # Msun/h
        'zMin_hmod': 0.005, 'zMax_hmod': 4,
        'zMin': 0, 'zMax': 2,
        'logM0': 0,

            }

    info['MhMin'] = cycle(info['MhMin'], lambda M, h=info['h']: M*u.Msun/h)
    info['MhMax'] = cycle(info['MhMax'], lambda M, h=info['h']: M*u.Msun/h)
    info['ndens'] = cycle(info['ndens'], lambda n: n/u.deg**2)




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
        'T_CMB': 2.725,  # mop-c-gt, mopc.py
        # galaxy catalog detailed, II.Ap1/Ap4/Figure2
        'MsMean': 3e11, 'MhMean': 3.3e13, # mean stellar and halo mass
        'zMin':0.4, 'zMax':0.7, 'z_med': 0.55,  # min/max/median redshift
        # Model implementation, Ip10, Appendix A p2
        'MassDef': 'fof', 'MassFunc':'sheth99', 'HaloBias':'sheth01',
    }
    
    info['T_CMB'] = cycle(info['T_CMB'], lambda T: T *u.K)
    info['MsMean'] = cycle(info['MsMean'], lambda M: M *u.Msun)
    info['MhMean'] = cycle(info['MhMean'], lambda M: M *u.Msun)
    
class Moser2021(BaseStudy):  # ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
    subs = {}
    info = Amodeo2021().info



class Koukoufilippas2020(BaseStudy):  # arxiv.org/abs/1909.09102
    subs={'sample':['2MPZ','WIxSC-1','WIxSC-2','WIxSC-3','WIxSC-4','WIxSC-5'],}
    info={}


class Naess2020(BaseStudy):  # ui.adsabs.harvard.edu/abs/2020JCAP...12..046N
    subs={}
    info={}


class Battaglia2016(BaseStudy):  # ui.adsabs.harvard.edu/abs/2016JCAP...08..058B
    subs={}
    info = {
        # sim's cosmo params, B15.3.P2
        'XH':0.76, 'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'h':0.72, 'ns':0.96, 'sigma8':0.8,
        'MassDef':'200c',  # Mass definition, B15.T2
        'MassFunc': 'Tinker08',
    }



class Kravtsov2018(BaseStudy):  # ui.adsabs.harvard.edu/abs/2018AstL...44....8K
    subs = {}
    info = {
        # fixed cosmo params, Section 1pLast
        'Om0':0.27, 'Ob0':0.0469, 'h':0.7, 'sigma8': 0.82, 'ns':0.95,
    }

class Vikram2017(BaseStudy):  # ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
    subs = {}
    info = {
        'ns':1, 'sigma8':0.8, 'Om0':0.27, 'Obl':0.73, 'Ob0':0.044, 'h':0.7,
        'MassDef':'200c', 'MassFunc':'Sheth99', 'HaloBias':'Sheth01',
    }


class More2015(BaseStudy):  # ui.adsabs.harvard.edu/abs/2015ApJ...806....2M
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




class Ahn2013Alam2015(BaseStudy):  # arxiv.org/abs/1501.00963, arxiv.org/abs/1307.7735
    subs = {'DR': ['DR10', 'DR12']
    }
    info = {
        'area': {'DR12': 9376, 'DR10': 6373.2},
        }

    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)

class Planck2013(BaseStudy):  # ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    subs = {}

    info = {'Om0':0.3, 'Ol0':0.7,'h':0.7,'MassDef':'500c',
        }


class Battaglia2011(BaseStudy):  # ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
    subs = {}
    info = {
        # Cosmological Parameters, 2p1/2p3/3p2
        'Om0':0.25, 'Ob0':0.043, 'Ol0':0.75, 'ns':0.96, 'sigma8':0.8, 'h':0.7, 'XH':0.76,
        'mdef':'200c',  # Mass definition, S2p3/Eq11
    }

class Arnaud2010(BaseStudy):  # ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
    subs = {}
    info = {
        # Cosmological Parameters, 1p-1
        'h':0.7, 'Om0':0.3, 'Ol0':0.7, 'Concentration': 'Constant', 'MassDef':'500c',
    }



class Nagai2007(BaseStudy):  # ui.adsabs.harvard.edu/abs/2007ApJ...668....1N
    subs = {}
    info = {
        # Cosmological Parameters, 2p1/2p3/3p2
        'Om0':0.3, 'Ob0':0.04286, 'h':0.7, 'sigma8':0.9,
        'Ol0':0.7, 'Fb':0.175,
        'MassDef':'500c', 'Concentration': 'Constant', # Mass definition, S2p3/Eq11
        # right after eq 1/2/3
        'mu':0.59, 'mu_e':'1.14'
    }
    
    
class Zheng2005(BaseStudy):  # ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
    subs = {}
    info = {
        }

class Zehavi2005(BaseStudy):  # ui.adsabs.harvard.edu/abs/2005ApJ...630....1Z
    subs = {}
    info = {
        }