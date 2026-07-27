"""
The Tenth Data Release of the Sloan Digital Sky Survey: First Spectroscopic Data from the SDSS-III Apache Point Observatory Galactic Evolution Experiment

ui.adsabs.harvard.edu/abs/2014ApJS..211...17A
arxiv.org/pdf/1307.7735
"""

from config import *
from Models.Papers.PlotsTables import BasePlots2
thispath = os.path.dirname(os.path.abspath(__file__))


class Data():
    # DR10 includes a total of 1,507,954 BOSS spectra, comprising 927,844 galaxy spectra; 182,009 quasar spectra; and 159,327 stellar spectra, selected over 6373.2 deg2
    area = 6373.2 * u.deg**2
    
    def __init__(self, inputdict={}, **inputvars):
        for key, value in (inputdict | inputvars).items(): setattr(self, key, value)
