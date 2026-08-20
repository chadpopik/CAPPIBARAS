"""
Atacama Cosmology Telescope: High-resolution component-separated maps across one third of the sky

ui.adsabs.harvard.edu/abs/2024PhRvD.109f3530C
arxiv.org/pdf/2307.01258
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable, read_wide_table
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Coulton2024")





"""Old implementation I'm phasing out"""




class MapData():  # ACT DR6 component-separated maps
    path = f"{DATA_PATH}/ACTDR6"  # Path to data downloaded from portal.nersc.gov/project/act/dr6_nilc/ymaps_20230220/
    # otherpath = "/global/cfs/projectdirs/act/data/act_dr6/dr6.02/beams/daytime_beams/"
    # NERSC_path = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220"  # path to data in NERSC
    subs={}
    info={}
    area = 12200*u.deg**2
    def __init__(self, inputsdict={}, **inputvars):
        ybeamdf = pd.read_csv(f"{self.path}/ilc_beam.txt", sep=" ")
        self.beam_ells, self.beam_data = ybeamdf['#'].values, ybeamdf['ell'].values
        self.resp_ells, self.resp_data = None, None

    def get_beam(self):
        return self.beam_ells, self.beam_data
