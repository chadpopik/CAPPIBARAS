"""
Missing baryons recovered: A measurement of the gas fraction in galaxies and groups with the kinematic Sunyaev-Zel'dovich effect and CMB lensing

ui.adsabs.harvard.edu/abs/2025PhRvD.112l3507H
arxiv.org/pdf/2507.14136
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, splittable, ParamTable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Hadzhiyska2025B")

from scipy.special import erf


    
def Cosmology():
    # = III.A As this analysis is performed at fixed cosmology, we employ the fiducial cosmology boxes which have cosmological parameters set to their Planck 2018 values: Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0.
    # info = {'h': 0.6736, 'MassDef': 'vir'
    # }
    # #Ω𝑏 ℎ2 = 0.02237, Ω𝑐 ℎ2 = 0.12,ℎ = 0.6736, 𝐴𝑠 = 2.0830 × 10−9, 𝑛𝑠 = 0.9649, 𝑤0 = −1,𝑤𝑎 = 0
    pass

def HaloModel():
    # III.A. All halo masses quoted in this work correspond to the mass definition adopted by the AbacusSummit halo finder CompaSO [46], which defines the virial mass using the spherical collapse model and the fitting formulae from Bryan and Norman [47].
    MassDef = 'vir'

# III.B. To model the distribution of Luminous Red Galaxies (LRGs) within dark matter halos, we adopt a standard (‘vanilla’) fiveparameter Halo Occupation Distribution (HOD) framework [Zheng et al 2005]. n this model, the mean number of central and satellite galaxies in a halo of mass 𝑀 is given by:
class HOD(): 
    MassDef = 'vir'
    def __init__(self, logMh, cosmology=None, halomodel=None, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
        
        self.cosmology = cosmology or Cosmology()
        self.halomodel = halomodel or HaloModel(self.cosmology)
        
        self.Mh = 10**logMh*u.Msun/cosmology.h
        self.logMh = np.log10(self.Mh/u.Msun)
        
        table = Table1_2_3()
        missing = False
        for col in ("Sample", "Bin"):  # validate each against the values in its table column
            options = table.df[col].dropna().unique().tolist()
            value = getattr(self, col, None)
            if value is None:
                raise ValueError(f"No {col} specified. Valid {col} options: {options}")
                # missing = True
            elif value not in options:
                raise ValueError(f"{col} '{value}' is not a valid option. Valid {col} options: {options}")
                # missing = True

        try: self.p0 = table.getparams(Sample=self.Sample, Bin=self.Bin).to_dict()
        except: self.p0 = {}
        
        if hasattr(self, "r"):
            from Models.Codes import mcfit
            fft = mcfit.FFT(rs=self.r)  # setup FFT
            self.k, self.FFT3D, self.IFFT3D = fft.ks, fft.FFT3D, fft.IFFT3D  # Define ks and FFT functions

        if hasattr(self, "k"):
            self.NFW = self.halomodel.NFW_k(k=self.k, z=self.z, logM=self.logMh)
            self.NFW_norm = self.NFW/self.NFW[0,:, :]
            self.dirac = np.ones_like(self.k)[:, None, None]

    def Ncen(self, pdict={}, **kwargs): # Eq 1
        p = self.p0 | pdict | kwargs
        return (1/2) * (1+erf((self.logMh-p['logMcut'])/(2*p['sigmalogM'])))

    def Nsat(self, pdict={}, **kwargs):  # Eq 3
        p = self.p0 | pdict | kwargs
        Mcut, M1 = 10**p['logMcut']*u.Msun, 10**p['logM1']*u.Msun
        return np.where(self.Mh>=p['kappa']*Mcut, ((self.Mh-p['kappa']*Mcut)/M1), 0)**p['alpha'] * self.Ncen(pdict, **kwargs)
    
    # NOTE: placeholder, needs normalization
    def ucen(self, pdict={}, **kwargs):
        return self.dirac
    
    # NOTE: placeholder
    def usat(self, pdict={}, **kwargs):
        return self.NFW_norm

    
# TABLE I. Best-fit values and 68% confidence intervals for the five HOD parameters and three derived parameters: comoving number density  ̄𝑛 (in [Mpc/ℎ]−3), satellite fraction 𝑓sat, and mean halo mass⟨𝑀halo⟩ (in 𝑀⊙ /ℎ). Results are shown for each of the three tracer samples: Main LRGs, Extended LRGs, and BGS. All mass units are in 𝑀⊙ /ℎ. The masses correspond to the virial mass definition from Bryan and Norman [47]. We budget around 7% for the systematic bias on the mean halo mass, as described in the main text.
# TABLE II. Best-fit HOD and derived parameters for the Main LRG sample, shown across four redshift bins. The redshift bins correspond to: Bin 1: 0.4, 0.54, 0.713, 0.86, 𝑧1 < 𝑧 < 0.54, Bin 2: 0.54 < 𝑧 <0.713, Bin 3: 0.713 < 𝑧 < 0.86, Bin 4: 0.86 < 𝑧 < 1.024. All mass units are in 𝑀⊙ /ℎ, and comoving number densities are in [Mpc/ℎ]−3. The masses correspond to the virial mass definition from Bryan and Norman [47]. We budget around 5% for the systematic bias on the mean halo mass (see Section III D).
# TABLE III. Same as Table II, but for the Extended LRG sample. The redshift bins are identical to those used in Table II.
class Table1_2_3(ParamTable):
    def __init__(self, filename=f"{thispath}/Table1_2_3.csv"):
        dfraw = pd.read_csv(filename)
        self.df, self.df_errlow, self.df_errhigh = splittable(dfraw)


class ParamsTable(ParamTable):  # characteristic HOD parameters, per mass bin
    def __init__(self, filename=f"{thispath}/params.csv"):
        self.df = read_wide_table(filename)

