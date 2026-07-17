"""
Galactic star formation and accretion histories from matching galaxies to dark matter haloes

ui.adsabs.harvard.edu/abs/2013MNRAS.428.3121M
arxiv.org/pdf/1205.5807
"""



import sys,os
from Models.Papers.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c