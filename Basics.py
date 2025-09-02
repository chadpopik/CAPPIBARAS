datapath = '/global/homes/c/cpopik/Data'
packpath = '/global/homes/c/cpopik/Capybara'
scratchpath = '/pscratch/sd/c/cpopik'
projectpath = '/global/cfs/projectdirs'

import numpy as np
import pandas as pd
import scipy, astropy, random
import os, sys, time, datetime
import h5py, pickle
import importlib
import subprocess
from typing import Optional, Sequence, Dict, Any

from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import astropy.constants as c
import astropy.cosmology.units as cu
from astropy.cosmology import default_cosmology, Planck18
default_cosmology.set(Planck18)

    
def README(dname):
    try: 
        strlist = ["README", "READ_ME", "readme", "read_me", "ReadMe", "Read_Me"]
        fname = [f"{dname}/{fname}" for fname in os.listdir(dname) if any(map(fname.__contains__, strlist))][0]
        print(open(fname, 'r').read())
    except: print("No README file")


def ViewDataOrganization(path, readme=True, files=False, levels=1, tab=0, mem=False):
    try:  # Access the folder or return a denial
        entries = os.listdir(path)
    except PermissionError:
        print(" "*tab+f"{path.split('/')[-1]}/ (Permission Denied)")
        return

    if readme:  # Print readme folder
        README(path)
        print('\n')

    if mem:
        memstr = f"{' '*5}({subprocess.run(['du', '-sm', path], capture_output=True, text=True).stdout.split()[0]} MB)"
    else:
        memstr=''
    if tab==0:  # Print base folder and its size
        print(f"{path}/{memstr}")
    else: 
        print(" "*tab+f"{path.split('/')[-1]}/{memstr}")
        
    if levels!=0:  # If going down more levels, then repeat the process
        for entry in entries:
            full_path = os.path.join(path, entry)  # Define path of entry in dir
            if os.path.isdir(full_path) is not True and (files is True or tab==0):  # If file, just print size
                memstr = subprocess.run(['du', '-sm', full_path], capture_output=True, text=True).stdout.split()[0]+' MB'
                print(" "*(tab+5)+f"{entry}{memstr}")
            elif os.path.isdir(full_path):  # If directory, repeat the process recursively
                ViewDataOrganization(full_path, readme=False, files=files, levels=levels-1, tab=tab+5)
        

# All the plotting stuff
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