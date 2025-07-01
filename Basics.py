homepath = '/global/homes/c/cpopik'
scratchpath = '/pscratch/sd/c/cpopik'
projectpath = '/global/cfs/projectdirs'

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import astropy, os, sys, h5py, time, random, scipy, pickle, h5py

import astropy.units as u
import astropy.constants as c
import astropy.cosmology.units as cu
from astropy.cosmology import default_cosmology, Planck18
default_cosmology.set(Planck18)


def README(dname):
    try: 
        strlist = ["README", "READ_ME", "readme", "read_me", "ReadMe", "Read_Me"]
        fname = [f"{dname}/{fname}" for fname in os.listdir(dname) if any(map(fname.__contains__, strlist))
][0]
        print(open(fname, 'r').read())
    except: print("No README file")


def ViewDataOrganization(path, readme=True, files=False, levels=1, tab=0):
    try: os.listdir(path)
    except PermissionError:
        print(" "*tab+f"{path.split('/')[-1]}/ (Permission Denied)")
        return
    if readme is True: README(path); print('\n')
    else: pass
    if tab==0: print(f"{path}/")
    else: print(" "*tab+f"{path.split('/')[-1]}/")
    if levels==0: return
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) is not True and (files is True or tab==0): print(" "*(tab+5)+f"{entry}")
        elif os.path.isdir(full_path): ViewDataOrganization(full_path, readme=False, files=files, levels=levels-1, tab=tab+5)  # Recursive call for subdirectories


import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
from matplotlib import gridspec

def setplot(dark=True):
    if dark is True: plt.style.use('dark_background')
    else: plt.style.use('default')
    plt.rcParams.update({
        'font.family':'serif', 'mathtext.fontset':'dejavuserif',
        'axes.grid':True, 'grid.linestyle': ':', 'grid.alpha': 0.5,
        'xtick.direction':'in', 'xtick.minor.visible': True, 'xtick.top':True,
        'ytick.direction':'in', 'ytick.minor.visible': True, 'ytick.right':True,
        'figure.figsize': [9, 6], 'axes.titlesize':30, 'legend.fontsize': 15, 'legend.title_fontsize': 20,
        'axes.labelsize':25, 'xtick.labelsize':15, 'ytick.labelsize':15
})