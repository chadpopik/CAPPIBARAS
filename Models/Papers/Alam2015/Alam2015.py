"""
The Eleventh and Twelfth Data Releases of the Sloan Digital Sky Survey: Final Data from SDSS-III


ui.adsabs.harvard.edu/abs/2015ApJS..219...12A
arxiv.org/pdf/1501.00963
"""

from config import *
from Models.Papers.PlotsTables import BasePlots2
thispath = os.path.dirname(os.path.abspath(__file__))

class Data():
    # The total footprint is about 10,400 deg2 (Figure 6); the value of 9376 deg2in Table 1 excludes masked regions due to bright stars and data that do not meet our survey requirements.
    area = 9376 * u.deg**2
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
