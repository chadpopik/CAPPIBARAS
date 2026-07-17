"""
Tomographic measurement of the intergalactic gas pressure through galaxy-tSZ cross-correlations


ui.adsabs.harvard.edu/abs/2020MNRAS.491.5464K
arxiv.org/pdf/1909.09102
"""


class Studies(BaseStudy):  # arxiv.org/abs/1909.09102
    subs={'sample':['2MPZ','WIxSC-1','WIxSC-2','WIxSC-3','WIxSC-4','WIxSC-5'],}
    info={}


class Measurements(BaseMeasurement, Studies.Koukoufilippas2020):  # arxiv.org/abs/1909.09102
    path = f"{datapath}/Koukoufilippas2020"  # path to data, taken from plots using webplotdigitizer
    subs = {'sample':['2MPZ','WIxSC-1','WIxSC-2','WIxSC-3','WIxSC-4','WIxSC-5']}

    def __init__(self, inputsdict, **inputvars):
        self.setup(inputsdict | inputvars)
        self.require(['sample'])

        self.get_meas()

    def get_meas(self):
        self.Cgy_data = self.Cgy[self.sample]
        self.Cgy_ell = self.ells[self.sample]

    def get_meas(self):        
        with h5py.File(f'{self.path}/Koukoufilippas2020_wpd.h5', "r") as f:
            self.Cgy_ell = f[f'ell_{self.sample}'][()]
            self.Cgy_data = f[f'Cgy_{self.sample}'][()]
