"""
Central location for filesystem paths and common imports shared across CAPPIBARAS.
"""

# List of common imports that are used throughout CAPPIBARAS, so that they can be imported from this single location. You probably should have all of these already
import numpy as np
import scipy
import pandas as pd
import time
import sys
import os
import h5py
import astropy
import astropy.units as u
import astropy.constants as c
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Plot Styling, this is all optional and can be overridden by the user in their own scripts
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family':'serif', 'mathtext.fontset':'dejavuserif',
    'axes.grid':True, 'grid.linestyle': ':', 'grid.alpha': 0.5,
    'xtick.direction':'in', 'xtick.minor.visible': True, 'xtick.top':True,
    'ytick.direction':'in', 'ytick.minor.visible': True, 'ytick.right':True,
})

# Root directory of CAPPIBARAS itself, derived from this file's location
CAPPIBARAS_PATH = Path(__file__).resolve().parent

# Path to data that couldn't fit in CAPPIBARAS and must be downloaded separately
DATA_PATH = Path("/global/homes/c/cpopik/Data")

# NOTE: You can ignore this
# Location of stacking/correlating results, used by the in-progress Popik2026 measurement/target data
STACKING_PATH = Path("/global/homes/c/cpopik/Stacking_Correlating")

# NOTE: you only need the following two if you're fitting things with cobaya
# Location of the SOLikeT package (added to sys.path since it's not pip-installed)
SOLIKET_PATH = Path("/global/homes/c/cpopik/soliket")

# Location where cobaya run outputs (chains, logs, etc.) are stored
OUTPUT_PATH = Path("/pscratch/sd/c/cpopik/CAPPIBARAS/runs")


__all__ = [
    "np",
    "pd",
    "plt",
    "astropy",
    "u",
    "c",
    "scipy",
    "os",
    "sys",
    "time",
    "Path",
    "cm",
    "h5py",
    "importlib",
    "CAPPIBARAS_PATH",
    "DATA_PATH",
    "SOLIKET_PATH",
    "OUTPUT_PATH",
    "STACKING_PATH",
]
