"""
Required/relevant information from studies used in this CAPPIBARAS, either for data, measurements, or models. Informaiton includes cosmological parameters, best-fit model parameters, properties of data, models used, etc.
"""


import astropy.units as u
import astropy.constants as c
import numpy as np



class BaseStudy:
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)

    def setup(self, inputs):
        self.check_inputs(inputdict=inputs, spefdict=self.subs)  # check inputs
        infospef = self.specify_info(self.info | self.params, inputdict=inputs)  # narrow down info using inputs
        for infokey, infoval in infospef.items(): setattr(self, infokey, infoval)  # set all info
        for inputkey, inputval in inputs.items(): setattr(self, inputkey, inputval)  # set remaining inputs, overriding info
        self.p0 = {p: getattr(self, p) for p in self.params.keys()}  # group all the fit parameters into a dict
        
        self.require = lambda reqlist: self.require_input(inputs, reqlist)
        
    def check_inputs(self, inputdict={}, spefdict={}, reqlist=[]):
        for spefkey, spefvals in spefdict.items():  # for all of the spefs that have lists
            if spefkey in inputdict.keys() and inputdict[spefkey] not in spefvals:  # if one of them is an input but the input isn't valid
                raise NameError(f"{spefkey} {inputdict[spefkey]} doesn't exist, choose from: {spefvals}")  # print an error
                
    def specify_info(self, infodict, inputdict={}):
        spefinfodict = infodict.copy()  # make a new dict
        for infokey in spefinfodict.keys():  # for all the general info and params
            for inp in inputdict.values(): # for every input 
                # NOTE: if an unrelated input has the same name as a spef, it will ALSO narrow this down. maybe fix this
                spefinfodict[infokey]=self.spefinfo(spefinfodict[infokey], inp)  # try to use it to narrow it down
        return spefinfodict

    def spefinfo(self, info, input):
        try: info = info[input]  # try to narrow down the current info input
        except: pass  # leave as is otherwise
        if not isinstance(info, dict): return info  # if it's not a dict, return the value
        return {k: self.spefinfo(v, input) for k, v in info.items()}  # otherwise recursively rebuild the dict

    def require_input(self, inputdict, reqlist):
        for req in reqlist: # for everything that's required
            if req not in inputdict.keys():  # if input doesn't have it
                raise ValueError(f"Missing {req} is required" )  # print an error
        
        
        
        
class Jenna_Catalog(BaseStudy):
    subs={}
    params={}
    info = {'area': 16700,  # assuming the same as XCorr LRGs
            }

                
                
class Popik2025(BaseStudy):  # In progress
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],
    }
    params = {
        
    }
    info = {
        
        }



class RiedGuachalla2025(BaseStudy):  # arxiv.org/abs/2503.19870
    subs = {'bin': ['all', 'z_1', 'z_2', 'z_3', 'z_4', 'mass_1', 'mass_2', 'mass_3', 'mass_4'],}  # subset of galaxy selection
    params = {}  # No fit parameters
    info = {
        'name': 'Ried-Guachalla 2025',
        'area': 4300,  # [deg^2] Figure 1
        'logmhalomean':13.4,  # min spec z, Section III.B p5
        'zmin': {'all':0.4, 'z_1':0.4, 'z_2':0.6, 'z_3':0.8, 'z_4':0.9},  # min spec z, Section III.B p6
        'zmax': {'all':1.1, 'z_1':0.6, 'z_2':0.8, 'z_3':0.95, 'z_4':1.1},  # max spec z, Section III.B p6
        'logmstarmin': {'all':10.5, 'mass_1':10.5, 'mass_2':11.2, 'mass_3':11.4, 'mass_4':11.6},  # Section III.B p7
        'logmstarmax': {'all':12.5, 'mass_1':11.2, 'mass_2':11.4, 'mass_3':11.6, 'mass_4':12.5},  # Section III.B p7
        'zmean': {'all':0.74, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.76, 'mass_2':0.75, 'mass_3':0.71, 'mass_4':0.69},  # mean z, Table II
        'zmed': {'all':0.75, 'z_1':0.51, 'z_2':0.71, 'z_3':0.87, 'z_4':1.01, 'mass_1':0.79, 'mass_2':0.76, 'mass_3':0.70, 'mass_4':0.67},  # median z, Table II
        'mstarmean': {'all':2.2, 'z_1':2.4, 'z_2':2.3, 'z_3':2.0, 'z_4':2.1, 'mass_1':1.2, 'mass_2':2.0, 'mass_3':3.0, 'mass_4':5.1},  # mean mstar [10e11 Mstar/Msun],   # Table II
        'ngal': {'all':825283, 'z_1':195877, 'z_2':235620, 'z_3':235620, 'z_4':96346, 'mass_1':244932, 'mass_2':320914, 'mass_3':194037, 'mass_4':53997},  # number of galaxies, Table II & Section III.B p6/p7
        }

    for bin in ['z_1', 'z_2', 'z_3', 'z_4']:  # assume z bins have same mass limits
        for val in ['logmstarmin','logmstarmax']: info[val][bin] = info[val]['all']
    for bin in ['mass_1', 'mass_2', 'mass_3', 'mass_4']:  # assume m bins have same z limits
        for val in ['zmin','zmax']: info[val][bin] = info[val]['all']
    for key, val in info['mstarmean'].items():  # Fix some units
        info['mstarmean'][key] = val * 1e11
    info['logmstarmean'] = {key: np.log10(m) for key, m in info['mstarmean'].items()}




class Hadzhiyska2025(BaseStudy):  # arxiv.org/abs/2407.07152
    subs = {'zbin': ['z1', 'z2', 'z3', 'z4'],}
    params={}
    info={}
    
#     # info = {'ngal': {'ext_DR9_z1': 963631, 'ext_DR9_z2': 1658313, 'ext_DR10_z3': 1951646, 'ext_DR10_z4':1690171, 'ext_all':6850072},
#     # # TODO: come back to this
#     # }
#     info = {'name':'Hadzhiyska 2025'}



class Liu2025(BaseStudy):  # arxiv.org/abs/2502.08850
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
    }
    params = {}  # No fit parameters
    info = {
        'name':'Liu 2025',
        'area': 7326 * u.deg**2,  # area of ACT DESI overlap, Fig 1
        'logmhalomean': {'z1':13.40, 'z2':13.40, 'z3':13.24, 'z4':13.24},  # rough mean halo mass, Section III.Ap5
        'zmean': {'z1':0.470, 'z2':0.628, '3':0.791, 'z4':0.924},  # objects in overlap, Table 1
        'ng': {'z1':81.9, 'z2':148.1, 'z3':162.4, 'z4':148.3},  # objects in overlap [deg^-2], Table 1
        'Ngal': {'z1':332280, 'z2':608100, 'z3':671738, 'z4':615543},  # objects in ACT DESI overlap, Table 1
        'Ngal_unmasked': {'z1':1118496, 'z2':2031303, 'z3':2240982, 'z4':2049158},  # total objects in entite sample, Table 1
    }
    
    for key in info['ng']: info['ng'][key] = info['ng'][key] / (u.deg**2)  # set right units
    
    
class Coulton2024(BaseStudy):  # arxiv.org/abs/2307.01258
    subs={}
    params={}
    info={}

        
class Zhou2023(BaseStudy):  # arxiv.org/abs/2309.06443
    subs = {
        'zbin' : ['z1', 'z2', 'z3', 'z4'],  # photometric redshift bin
        'sample' : ['main', 'ext'],  # sample of LRGs
        'hemisphere' : ['combined', 'north', 'south'],  # sky hemisphere
    }
    params = {}  # No fit parameters
    info = {
        'name': 'Zhou 2023',
        'area': {  # area of survey [deg^2]], Section 2.1p2/3.2p3
                'main':{'combined':16700, 'north':4200, 'south':12500},
                'extended':{'combined':230, 'north':100, 'south':130}},
        'ndens': {  # number density [deg^-2], Table 1/2/3
            'main': {'all':600, 'z1': 81.9, 'z2': 148.1, 'z3': 162.4, 'z4': 148.3},
            'ext': {'all':1669, 'z1': 185.5, 'z2': 311.0, 'z3': 422.6, 'z4': 438.4},},
        'zmean': {   # mean redshift, Table 2/3
            'main': {'z1': 0.470, 'z2': 0.628, 'z3': 0.791, 'z4': 0.924},
            'ext': {'z1': 0.467, 'z2': 0.633, 'z3': 0.794, 'z4': 0.929},},
        'logmhalomean': {'z1': 13.40, 'z2': 13.40, 'z3': 13.24, 'z4': 13.24},  # mean halo mass from HOD study [Msun?], Section 6p2
        'pzmin': {  # min photometric redshift, Table 2/3
            'main': {
                'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, 'z3': 0.719, 'z4': 0.851},
                'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, 'z3': 0.713, 'z4': 0.860},},
            'ext': {
                'north': {'all':0.400, 'z1': 0.400, 'z2': 0.545, '3': 0.719, 'z4': 0.854},
                'south': {'all':0.400, 'z1': 0.400, 'z2': 0.540, '3': 0.713, 'z4': 0.860},},},
        'pzmax': {  # max photometric redshift, Table 2/3
            'main': {
                'north': {'all':1.024, 'z1': 0.545, 'z2': 0.719, 'z3': 0.851, 'z4': 1.024},
                'south': {'all':1.020, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.020},},
            'ext': {
                'north': {'all':1.010, 'z1': 0.545, 'z2': 0.719, 'z3': 0.854, 'z4': 1.010},
                'south': {'all':1.000, 'z1': 0.540, 'z2': 0.713, 'z3': 0.860, 'z4': 1.000},},},
    }
    for val in ['pzmin', 'pzmax']:  # Assumeding combined uses south limits
        for samp in info[val].keys(): 
            info[val][samp]['combined'] = info[val][samp]['south']
            
            
class Gao2023(BaseStudy):  # arxiv.org/abs/2306.06317
    subs = {'model':["ELG", "ELGX", "Psat"],}
    params = {
        'logM0': {'ELG': 11.56, 'ELGX': 12.14, 'Psat': 12.07},
        'alpha': {'ELG': 0.43,  'ELGX': 0.37,  'Psat': 0.37},
        'beta': {'ELG': 2.72,  'ELGX': 2.27,  'Psat': 2.61},
        'logk': {'ELG': 10.11, 'ELGX': 10.40, 'Psat': 10.36},
        'sigma': {'ELG': 0.18,  'ELGX': 0.21,  'Psat': 0.21},
    }
    info = {'mdef':'vir',  # Current Virial Mass
            'area': 140,  # covering 20 separate ”rosette” areas, each of which is approximately 7 deg2.
            'h':0.71, 'Omega_m':0.268, 'Omega_L':0.732,
            }


class Yuan2023(BaseStudy):  # arxiv.org/abs/2306.06314
    subs = {'zbin': ['LRG1', 'LRG2', 'QSO', 'LRG3', 'LRG4']}
    info = {
        'mdef': '200c',  # M not clear, maybe same as zheng 2005/2007? or cmass?
        'mhalomin': 1.3e11,  # Msun/h
        'zmin': {'LRG1': 0.6, 'LRG2': 0.8, 'QSO': 2.1, 'LRG3': 0.95, 'LRG4': 0.8},
        'zmax': {'LRG1': 0.4, 'LRG2': 0.6, 'QSO': 0.8, 'LRG3': 0.8, 'LRG4': 0.95},
    }
    params = {
        "logM_cut": {"LRG1": 12.89, "LRG2": 12.78, "QSO": 12.67, "LRG3": 12.89, "LRG4": 12.68}, # Msun/h
        "logM_1": {"LRG1": 14.08, "LRG2": 13.94, "QSO": 15.00, "LRG3": 13.96, "LRG4": 13.60}, # Msun/h
        "sigma": {"LRG1": 0.27, "LRG2": 0.23, "QSO": 0.58, "LRG3": 0.37, "LRG4": 0.53},
        "alpha": {"LRG1": 1.20, "LRG2": 1.07, "QSO": 1.09, "LRG3": 0.91, "LRG4": 0.72},
        "kappa": {"LRG1": 0.65, "LRG2": 0.55, "QSO": 0.74, "LRG3": 0.74, "LRG4": 0.51},
        "f_ic": {"LRG1": 0.92, "LRG2": 0.89, "QSO": 0.041, "LRG3": 0.92, "LRG4": 0.19},
        "f_sat": {"LRG1": 0.089, "LRG2": 0.104, "QSO": 0.05, "LRG3": 0.110, "LRG4": 0.151},
        "logM_h_mean": {"LRG1": 13.42, "LRG2": 13.26, "QSO": 12.74, "LRG3": 13.29, "LRG4": 13.00},
        "b_lin": {"LRG1": 1.94, "LRG2": 2.11, "QSO": 2.56, "LRG3": 2.31, "LRG4": 2.13},
    }



class Xu2023(BaseStudy):  # arxiv.org/abs/2211.02665
    subs = {'sample': ['Main', 'LOWZ', 'CMASS'],
            'form': ['BP13', 'DP']}
    params = {
        "logM0": {
            "BP13": {"Main": 11.338, "LOWZ": 11.359, "CMASS": 11.509},
            "DP":   {"Main": 11.732, "LOWZ": 11.579, "CMASS": 11.624},},
        "alpha": {
            "BP13": {"Main": 0.484, "LOWZ": 0.623, "CMASS": 0.740},
            "DP":   {"Main": 0.299, "LOWZ": 0.429, "CMASS": 0.466},},
        "delta": {
            "BP13": {"Main": 3.041, "LOWZ": 3.248, "CMASS": 2.964},},
        "beta": {
            "BP13": {"Main": 1.632, "LOWZ": 1.702, "CMASS": 2.094},
            "DP":   {"Main": 1.917, "LOWZ": 2.215, "CMASS": 2.513},},
        "logeps": {
            "BP13": {"Main": -1.545, "LOWZ": -1.598, "CMASS": -1.565},},
        "logk": {
            "DP":   {"Main": 10.303, "LOWZ": 10.105, "CMASS": 10.133},},
        "sigma": {
            "BP13": {"Main": 0.237, "LOWZ": 0.190, "CMASS": 0.190},
            "DP":   {"Main": 0.233, "LOWZ": 0.201, "CMASS": 0.192},}
    }
    info = {
        'mdef': 'vir',  # virial mass of the halo at the time when the galaxy was last the central dominant object
    }
            


class Kou2023(BaseStudy):  # arxiv.org/abs/2211.07502
    subs = {'mbin':['M1', "M2", "M3", "M4"],}
    params = {
        "logM_min": {"M1": 13.47, "M2": 13.58, "M3": 13.84, "M4": 14.20},  # minimum halo mass for a central galaxy/halos contain 0.5 central galaxies on average
        "sigma_logM": {"M1": 0.76, "M2": 0.78, "M3": 0.86, "M4": 0.959},  # changes the number of galaxies in low-mass halos
        "logM_1": {"M1": 14.119, "M2": 14.140, "M3": 14.171, "M4": 14.100},  # controls the number of galaxies at high halo mass
        "beta_s": {"M1": 4.38, "M2": 4.71, "M3": 5.31, "M4": 6.35},  # satellite galaxy profile
        "1-b_h": {"M1": 0.602, "M2": 0.623, "M3": 0.558, "M4": 0.550},  # hydrostatic bias
        "A": {"M1": 0.981, "M2": 0.965, "M3": 0.956, "M4": 0.961},  # cross-correlation amplitude
        "alpha_inc": {"M1": 0.51, "M2": 0.42, "M3": 0.39, "M4": 0.33},  # included to account for galaxy incompleteness at the low stellar mass end
        "logM_inc": {"M1": 13.39, "M2": 13.42, "M3": 13.69, "M4": 13.96},  # included to account for galaxy incompleteness at the low stellar mass end
        "beta_m": {"M1": 4.97, "M2": 5.91, "M3": 4.16, "M4": 10},  # matter density profile
    }
    info = {
        'mdef': '200m',  # region in which the average density is ∆ = 200 times the cosmic mean density
        'zlims': [0.47, 0.59],  # redshift range of selected galaxies
        'zmed': 0.53,  # median redshift
        'logmstarmin': {'M1': 10.8, 'M2': 11.1, 'M3': 11.25, 'M4': 11.4},  # minimum stellar mass of selected
        'H0':67.66, 'Omega_bh2':0.02242, 'Omega_ch2':0.11933, 'tau':0.0561, 'n_s':0.9665, 'sigma8':0.8102, # 5.1p3
    }
    info['h'] = info['H0']/100
    info['Omega_b'], info['Omega_c'] = info['Omega_bh2']/info['h']**2, info['Omega_ch2']/info['h']**2
    info['Omega_c'] = info['Omega_ch2']/info['h']**2
    info['Omega_m'] = info['Omega_c'] + info['Omega_b']



class Linke2022(BaseStudy):  # arxiv.org/abs/2204.02418
    subs = {'sample':["MSr", "MSb", "KVGr", "KVGb"]}  # TODO: There are actually many further subsamples cut by stellar mass
    params = {
        "alpha^a": {"MSr": 0.47, "MSb": 0.10, "KVGr": 0.34, "KVGb": 0.13},
        "sigma^a": {"MSr": 0.55, "MSb": 0.47, "KVGr": 0.52, "KVGb": 0.47},
        "M_th^a": {"MSr": 23.0, "MSb": 1.19, "KVGr": 15, "KVGb": 1.4},  # units of 1e11 Msol
        "beta^a": {"MSr": 0.84, "MSb": 0.73, "KVGr": 0.88, "KVGb": 0.55},
        "M^a": {"MSr": 5.8, "MSb": 32, "KVGr": 3.6, "KVGb": 20},       # units of 1e13 Msol
        "f": {"MSr": 1.49, "MSb": 0.88, "KVGr": 1.27, "KVGb": 0.83},
        "A": {"MSr": 5.31, "MSb": 5.31, "KVGr": 1.62, "KVGb": 1.62},
        "epsilon": {"MSr": 0.69, "MSb": 0.69, "KVGr": 0.99, "KVGb": 0.99},
    }
    info = {'mdef': '200m', 'zmax': 0.5,
            'mhalomin':10e11, 'mhalomax': 10e15,  # Msun/h^2, these cuts are just for sims
    }


class White2022(BaseStudy):  # arxiv.org/abs/2111.09898
    subs = {'zbin' : ['z1', 'z2', 'z3', 'z4'],} # photometric redshift subsmaple
    params = {}
    info = {
        'name':'White 2022',
        'area': 18000,  # area of DESI survey, [deg^2], Figure 1
        'zmean': {'z1':0.47, 'z2':0.63, 'z3':0.79, 'z4':0.92},  # mean redshift, Table 1
        'ndens': {'z1':83, 'z2':149, 'z3':162, 'z4':149},  # galaxy number desnity, Table 1
        'SN': {'z1':4.02, 'z2':2.24, 'z3':2.07, 'z4':2.26},  # shot noise level [1e6], Table 1
        'zeff': {'z1':0.47, 'z2':0.62, 'z3':0.78, 'z4':0.91},  # effective z at which power spec is calculate approx, Table 1
        'lmax': {'z1':250, 'z2':300, 'z3':350, 'z4':400},  # max ell used in fits, Table 1
        'lSN': {'z1':400, 'z2':530, 'z3':575, 'z4':425},  # ell where SN equals modeled auto-spec power, Table 1
    }



class Kusiak2022(BaseStudy):  # arxiv.org/abs/2203.12583
    subs = {'sample':['Blue', 'Green', 'Red'],}
    params = {
        "sigma_logM": {"Blue": 0.73, "Green": 0.61, "Red": 0.75},
        "alpha_s": {"Blue": 1.38, "Green": 1.23, "Red": 1.18},
        "logM_min^HOD": {"Blue": 12.11, "Green": 12.39, "Red": 13.23},
        "logM_1": {"Blue": 13.00, "Green": 12.87, "Red": 12.20},
        "lambda": {"Blue": 1.11, "Green": 2.50, "Red": 1.30},
        "10^7A_SN": {"Blue": -0.16, "Green": 1.35, "Red": 27.95},
    }
    info = {'mdef': '200c', 
        'mhalomin': 7e8, 'mhalomax': 3.5e15,  # Msun/h
        'zmin_hmod': 0.005, 'zmax_hmod': 4,
        'zmin': 0, 'zmax': 2,
        'zmean': {'Blue': 0.6, 'Green': 1.1, 'Red': 1.5},
        'logM0': 0,
        'omega_c': 0.11933, 'omega_b': 0.02242, 'H0':67.66, 'n_s':0.9665, 'lnAsn10': 3.047, 'kpivot':0.05,'tau_reio':0.0561,  # Ip7
        'Concentration': 'Bhattacharya13',
            }
    info['h'] = info['H0']/100
    info['Omega_c'], info['Omega_b'] = info['omega_c']/info['h']**2, info['omega_b']/info['h']**2
    info['Omega_m'] = info['Omega_c'] + info['Omega_b']
    

        
class Schaan2021(BaseStudy):  # arxiv.org/abs/2009.05557
    subs = {
        'sample' : ['cmass', 'lowz'],  # galaxy sample (CMASS M from DR12 not available for everything)
    }
    params = {}  # No fit parameters
    info = {
        'name':'Schaan 2021',
        'area': 6000,  # area of overlap between ACT and BOSS [deg^2], TODO 1: assumed
        'mdef':'vir', 'mhalomean': {'lowz':5e13, 'cmass':3e13},  # halo mass definition and mean halo masses, Figure 3
        'mstarmax': 5.5e11, 'mhalomax': 1e14, # max stellar mass and halo mass, Section IV.Ep2
        'zmin':0.4, 'zmax':0.7,  # redshift range, Section IIp1
        'zmean': {'lowz':0.31, 'cmass':0.55},  # mean redshift, Figure 2 (says 0.55 everywhere else in the paper)
        'ngal_catalog':{'lowz':218905, 'cmass': 501844, 'CMASSm':777202},  # total galaxies in BOSS catalog, Section III.Ap2
        'ngal_overlap': {'lowz':151713, 'cmass': 325518, 'CMASSm':385137},  # galaxies in ACT BOSS overlap, Section III.Ap2
        'ngal_masked': {'lowz':145714, 'cmass': 312708, 'CMASSm':368701},  # galaxies in overlap after masking, Section III.Ap2
        'ngal': {'lowz':134702, 'cmass':311309, 'CMASSm':360084},  # final galaxy count after applying upper mass limit, Section III.Ap2
        'T_CMB':2.726, # CMB temp [K], Section F.1p8
        'v_rms': {'lowz':320, 'cmass':313},  # rms velocity [km/s] at mean redshifts, Section F.1p8/F.2p1
        }
    for val in ['mstarmax', 'mhalomax']:  # get log values for masses
        info[f'log{val}'] = np.log10(info[val])
    info['logmhalomean'] = {key: np.log10(val) for key, val in info['mhalomean'].items()}



class Amodeo2021(BaseStudy):  # arxiv.org/abs/2009.05558
    subs = {'model': ['GNFW', 'OBB', 'DustH', 'DustAH'],}  # pres/dens profile model
    params = {  # free params, all from Table II
        'logrho0': {'GNFW': 2.6},  # density log amplitude
        'xc_k': {'GNFW': 0.6},     # density core radius
        'beta_k': {'GNFW': 2.6},   # density outer slope
        'A2h_k': {'GNFW': 1.1},    # density 2h amplitude
        'P0': {'GNFW': 2.0},       # pressure amplitude
        'alpha_t': {'GNFW': 0.8},  # pressure intermediate slope
        'beta_t': {'GNFW': 2.6},   # pressure outer slope
        'A2h_t': {'GNFW': 0.7},    # pressure 2h amplitude
        
        'A_dust': {'DustH': 0.326, 'DustAH': 0.363},  # amplitude of dust emission [kJy/sr]
        'T_dust': {'DustH': 20.7,  'DustAH': 16.9},   # Dust temperature [K]
        'beta_dust':{'DustH': 1.13, 'DustAH': 1.13},  # Dust spectral index
        'c_0':      {'DustH': 5.00,'DustAH': 6.046},  # Polynomial coefficient on x^0
        'c_1':      {'DustH': -1.48, 'DustAH': -1.88}, # Polynomial coefficient on x^1
        'c_2':      {'DustH': 0.113, 'DustAH': 0.148}, # Polynomial coefficient on x^2
    }
    info = {'Omega_m': 0.25, 'Omega_b': 0.044, 'Omega_L': 0.75, 'H0': 70, 'mdef': '200c',  # cosmological parameters, Section Ip10
            'v_rms': 1.06e-3, 'X_H':0.76,   # RMS of peculiar velocites [v/c] and hydrogen mass fraction, Section IIA.Ap4/p5
            'mstar_mean': 3e11,  # mean stellar mass, Section II.Ap1
            'mhalo_mean': 3.3e13,  # mean halo mass, Section III.Ap4/Figure 2
            'zmin':0.4, 'zmax':0.7, 'z_med': 0.55,  # min/max/median redshift, Section II.Ap1
            'logmhalomin_2h':10, 'logmhalomax_2h':15,  # mass range used in two-halo integral, Appendix A p2
            'dndlogm_mod':'ST02', 'bh_mod':'ST01',  # Appendix A p2
            # fixed params
            'gamma_t': -0.3, 'xc_t_A0':0.497, 'xc_t_alpham': -0.00865, 'xc_t_alphaz':0.731,  # fixed GNFW pres params, Section II.Cp3
            'gamma_k': -0.2, 'alpha_k':1,  # fixed GNFW dens params, Section II.Cp2
            'z0': 0.55,  # redshift of the dust emitters, II.A.p1
            'nu0': ((c.c/(350*u.um)).to(u.GHz)).value,  # rest-frame frequency at which we normalize the dust emission, assumed from matching Fig11 I(v) plots
            'T_CMB': 2.725,  # mop-c-gt, mopc.py
            }
    info['h'] = info['H0']/100
    info['Omega_c'] = info['Omega_m']-info['Omega_b']
    info['f_bary'] = info['Omega_b']/info['Omega_m']



class Koukoufilippas2020(BaseStudy):  # arxiv.org/abs/1909.09102
    subs={'sample':['2MPZ','WIxSC-1','WIxSC-2','WIxSC-3','WIxSC-4','WIxSC-5'],}
    params={}
    info={}


class Naess2020(BaseStudy):  # arxiv.org/abs/2007.07290
    subs={}
    params={}
    info={}


class Battaglia2018(BaseStudy):  # arxiv.org/abs/1607.02442
    subs = {'model':['AGN', 'SH'],}  # AGN feedback vs shock heating sub-grid physics models
    params = {
        'rho0_A0': {'AGN': 4e3, 'SH': 1.9e4},   # density amplitude
        'rho0_alpham': {'AGN': 0.29, 'SH': 0.09},
        'rho0_alphaz': {'AGN': -0.66, 'SH': -0.95},
        'alpha_A0': {'AGN': 0.88, 'SH': 0.70},   # density intermediate slope
        'alpha_alpham': {'AGN': -0.03, 'SH': -0.017},
        'alpha_alphaz': {'AGN': 0.19, 'SH': 0.27},
        'beta_A0': {'AGN': 3.83, 'SH': 4.43}, # density asymptotic fall off power law index
        'beta_alpham': {'AGN': 0.04, 'SH': 0.005},
        'beta_alphaz': {'AGN': -0.025, 'SH': 0.037},
    }
    info = {
        'X_H':0.76, 'Omega_m':0.25, 'Omega_b':0.043, 'Omega_L':0.75, 'H0':72, 'n_s':0.96, 'sigma8':0.8,  # sim's cosmo params, B15.3.P2
        'h':0.72,
        'mdef':'200c',  # Mass definition, B15.T2
        'xc': 0.5 , 'gamma': -0.2,  # fixed GNFW params, B15.A.P2
    }
    info['f_bary'] = info['Omega_b']/info['Omega_m']


class Kravstov2018(BaseStudy):  # arxiv.org/abs/1401.7329
    subs = {'mdef': ["m200c", "m500c", "m200m", "mvir"],
            'scatter': ['B', 'S']}
    params = {
        "logM1": {
            "B": {"m200c": 11.39, "m500c": 11.32, "m200m": 11.45, "mvir": 11.43},
            "S": {"m200c": 11.35, "m500c": 11.28, "m200m": 11.41, "mvir": 11.39},},
        "logeps": {
            "B": {"m200c": -1.618, "m500c": -1.527, "m200m": -1.702, "mvir": -1.663},
            "S": {"m200c": -1.642, "m500c": -1.556, "m200m": -1.720, "mvir": -1.685},},
        "alpha": {
            "B": {"m200c": 1.795, "m500c": 1.856, "m200m": 1.736, "mvir": 1.750},
            "S": {"m200c": 1.779, "m500c": 1.835, "m200m": 1.727, "mvir": 1.740},},
        "delta": {
            "B": {"m200c": 4.345, "m500c": 4.376, "m200m": 4.273, "mvir": 4.290},
            "S": {"m200c": 4.394, "m500c": 4.437, "m200m": 4.305, "mvir": 4.335},},
        "gamma": {
            "B": {"m200c": 0.619, "m500c": 0.644, "m200m": 0.613, "mvir": 0.595},
            "S": {"m200c": 0.547, "m500c": 0.567, "m200m": 0.544, "mvir": 0.531},},
    }
    info={}
    

    
class More2015(BaseStudy):  # arxiv.org/abs/1407.1856 
    subs = {'mbin': ['M1', 'M2', 'M3']}
    params = {
        "logM_min": {"M1": 13.13, "M2": 13.45, "M3": 13.68},
        "sigma^2": {"M1": 0.22, "M2": 0.45, "M3": 0.79},
        "logM_1": {"M1": 14.21, "M2": 14.51, "M3": 14.56},
        "alpha": {"M1": 1.13, "M2": 1.14, "M3": 1.00},
        "kappa": {"M1": 1.25, "M2": 0.85, "M3": 1.19},
        "M_stellar_11": {"M1": 0, "M2": 0, "M3": 0},  # units of 10^11 h^(-2) M_solar
        "R_c": {"M1": 0.98, "M2": 1.01, "M3": 1.02},
        "psi": {"M1": 0.93, "M2": 0.93, "M3": 0.94},
        "p_off": {"M1": 0.34, "M2": 0.37, "M3": 0.36},
        "R_off": {"M1": 2.2, "M2": 2.3, "M3": 2.4},
        "alpha_inc": {"M1": 0.44, "M2": 0.53, "M3": 0.57},
        "logM_inc": {"M1": 13.57, "M2": 13.88, "M3": 14.08},
        "Omega_m": {"M1": 0.310, "M2": 0.306, "M3": 0.304},
        "sigma_8": {"M1": 0.785, "M2": 0.839, "M3": 0.813},
        "100*Omega_b*h^2": {"M1": 2.228, "M2": 2.226, "M3": 2.222},
        "n_s": {"M1": 0.964, "M2": 0.963, "M3": 0.961},
        "h": {"M1": 0.703, "M2": 0.700, "M3": 0.695},
    }
    info = {
        'mdef': '200m',  # M200b, 200 times overdense wrt background matter density
        'logmstarmin': {"M1": 11.10, "M2": 11.30, "M3": 11.40},
        'logmstarmax': {"M1": 12.00, "M2": 12.0, "M3": 12.0},
    }
    
    
class Ahn2013Alam2015(BaseStudy):  # arxiv.org/abs/1501.00963, arxiv.org/abs/1307.7735
    subs = {'DR': ['DR10', 'DR12']
    }
    params={}
    info = {
        'area': {'DR12': 9376, 'DR10': 6373.2},
        }
    
 

class Battaglia2011(BaseStudy):  # arxiv.org/abs/1109.3711
    subs = {}  
    params = {  # all from T1
        'P0_A0': 18.1, 'P0_alpham': 0.154, 'P0_alphaz': -0.758,  # Amplitude
        'xc_A0': 0.497, 'xc_alpham': -0.00865, 'xc_alphaz': 0.731,  # Core-scale
        'beta_pres_A0': 4.35,  'beta_pres_alpham': 0.0393,  'beta_pres_alphaz': 0.415,  # Asymptotic fall off power law index
    }
    info = {
        'X_H':0.76, 'Omega_m':0.25, 'Omega_b':0.043, 'Omega_L':0.75, 'n_s':0.96, 'sigma8':0.8, 'h':0.7,  # Sim's cosmo params, S2p1/S2p3
        'mdef':'200c',  # Mass definition, S2p3/Eq11
        'alpha_pres': 1, 'gamma_pres': -0.3,  # Fixed GNFW params, S4.1p1
    }
    info['f_bary'] = info['Omega_b']/info['Omega_m']
    