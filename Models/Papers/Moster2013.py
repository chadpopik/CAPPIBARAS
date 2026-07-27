"""
Galactic star formation and accretion histories from matching galaxies to dark matter haloes

ui.adsabs.harvard.edu/abs/2013MNRAS.428.3121M
arxiv.org/pdf/1205.5807
"""


from config import *
from Models.Papers.Figures.PlotsTables import BasePlots2, ParamTable, splittable
thispath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "Moster2013")





"""Old Implementation being phased out"""

from Models.Studies import BaseStudy, cycle
class Study(BaseStudy): 
    pass
