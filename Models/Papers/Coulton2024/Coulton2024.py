"""
Atacama Cosmology Telescope: High-resolution component-separated maps across one third of the sky

ui.adsabs.harvard.edu/abs/2024PhRvD.109f3530C
arxiv.org/pdf/2307.01258
"""


class Studies(BaseStudy):  # ui.adsabs.harvard.edu/abs/2024PhRvD.109f3530C
    subs={}
    info={'area': 12200
          }
    info['area'] = cycle(info['area'], lambda a: a*u.deg**2)


class MapData(BaseMapData, Studies.Coulton2024):  # ACT DR6 component-separated maps
    path = f"{datapath}/ACTDR6"  # Path to data downloaded from portal.nersc.gov/project/act/dr6_nilc/ymaps_20230220/
    # otherpath = "/global/cfs/projectdirs/act/data/act_dr6/dr6.02/beams/daytime_beams/"
    # NERSC_path = "/global/cfs/projectdirs/act/www/dr6_nilc/ymaps_20230220"  # path to data in NERSC
    subs={}
    info={}
    def __init__(self, inputsdict={}, **inputvars):
        self.setup(inputsdict | inputvars)
        ybeamdf = pd.read_csv(f"{self.path}/ilc_beam.txt", sep=" ")
        self.beam_ells, self.beam_data = ybeamdf['#'].values, ybeamdf['ell'].values
        self.resp_ells, self.resp_data = None, None

    def get_beam(self):
        return self.beam_ells, self.beam_data
