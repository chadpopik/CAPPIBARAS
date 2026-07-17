"""
The Tenth Data Release of the Sloan Digital Sky Survey: First Spectroscopic Data from the SDSS-III Apache Point Observatory Galactic Evolution Experiment

ui.adsabs.harvard.edu/abs/2014ApJS..211...17A
arxiv.org/pdf/1307.7735
"""

import sys,os
from Models.Papers.PlotsTables import BasePlots2
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c

    # subs = {'DR': ['DR10', 'DR12']
    # }
    # info = {
    #     'area': {'DR12': 9376, 'DR10': 6373.2},
    #     }

    # info['area'] = cycle(info['area'], lambda a: a *u.deg**2)


class Studies(BaseStudy):  # The Tenth Data Release of the Sloan Digital Sky Survey: First Spectroscopic Data from the SDSS-III Apache Point Observatory Galactic Evolution Experiment, ui.adsabs.harvard.edu/abs/2014ApJS..211...17A
    subs = {'DR': ['DR10', 'DR12']
    }
    info = {
        'area': {'DR12': 9376, 'DR10': 6373.2},
        }

    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)
