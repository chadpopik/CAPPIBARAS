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
import json
import h5py
import astropy
import astropy.units as u
import astropy.constants as c
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

    # def __init__(self, inputdict={}, **inputvars):
    #     for key, value in (inputdict | inputvars).items(): setattr(self, key, value)

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

# Machine-local paths (DATA_PATH, SOLIKET_PATH, OUTPUT_PATH) live in
# config_local.py, which is gitignored so each checkout can set its own without churn.
# DATA_PATH: location of any and all data used in the forward model (catalogs, measurements, target distributions, etc)
# SOLIKET_PATH: location of the SOLikeT package
# OUTPUT_PATH: location to put the output of cobaya jobs
from config_local import DATA_PATH, SOLIKET_PATH, OUTPUT_PATH


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
    "json",
    "importlib",
    "CAPPIBARAS_PATH",
    "DATA_PATH",
    "SOLIKET_PATH",
    "OUTPUT_PATH",
]
