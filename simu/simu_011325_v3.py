# Initialize notebook 

# %matplotlib notebook
%matplotlib widget

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import RectangleSelector
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import scipy.io as si
import scipy.signal as ss
from scipy.interpolate import RegularGridInterpolator as RGI
from time import time, localtime, strftime
from datetime import date
import gdspy
from shapely import Polygon
from PIL import Image
from imageio import imread as imreader

import tifffile as tif
from xrayutilities.io import spec
# import cmath
from matplotlib import cm
import imutils as imu
from PIL import Image as pim

import h5py as h5
from skimage.restoration import unwrap_phase

# from xraylib import *
import xraylib as xb

from multiprocessing import Pool, cpu_count
from functools import partial
from function_test import *

from IPython.display import display

from ipywidgets import interact,widgets,Box

import sys
sys.path.append('/nsls2/users/ccheu/')
from Functions_File_Operation import *
from Functions_General_Geometry import *
from Functions_General_Algebra import *
from Functions_BCDI_DataProc import *
from Functions_Crystallography import *

# del Pool, cpu_count

def gaussplots(*args):
    # Gaussian wave at focal plane 
    flag_plot = True

    x_ax = (np.arange(wave_prop_size[0]) - int(wave_prop_size[0]/2)) * wave_grid_step[0]
    y_ax = (np.arange(wave_prop_size[1]) - int(wave_prop_size[1]/2)) * wave_grid_step[1] 
    # amplitude
    wf_fc_amp = Gaussian_2D(x_ax,y_ax,1,0, gau_cen_offset[0], gau_width_param[0]/2.355, gau_cen_offset[1],gau_width_param[1]/2.355)
    # phase
    wf_fc_ph = np.zeros(wave_prop_size)
    wf_fc = wf_fc_amp * np.exp(-1j * wf_fc_ph)
    # wavefront at focal plane
    print(np.shape(wf_fc))
    wf_fc *= np.sqrt(beam_flux / np.sum( np.abs(wf_fc * np.conjugate(wf_fc))))
    print('%g' %np.sum( np.abs(wf_fc * np.conjugate(wf_fc)) ) )
    if flag_plot:   # 1D plot of wavefront 
        temp = wf_fc
        lim_region = False
        plot_region = [-10, 10]

        a = plt.figure(figsize=(10,3.5))
        plt.subplot(121)
        plt.plot(x_ax/1e3, np.abs(temp)[:, int(np.shape(temp)[1]/2 + gau_cen_offset[1]/wave_grid_step[1])], label='x amp')
        plt.plot(y_ax/1e3, np.abs(temp)[int(np.shape(temp)[0]/2 + gau_cen_offset[0]/wave_grid_step[0]), :], label='y amp')
        plt.legend(); plt.xlabel('x/y_ax [um]')
        if lim_region: 
            plt.xlim(plot_region)
        plt.subplot(122)
        plt.plot(x_ax/1e3, np.angle(temp)[:, int(np.shape(temp)[1]/2 + gau_cen_offset[1]/wave_grid_step[1])], label='x ph')
        plt.plot(y_ax/1e3, np.angle(temp)[int(np.shape(temp)[0]/2 + gau_cen_offset[0]/wave_grid_step[0]), :], label='y ph')
        plt.legend(); plt.xlabel('x/y_ax [um]')
        if lim_region: 
            plt.xlim(plot_region)
        plt.tight_layout()
        del temp, plot_region, lim_region
    if flag_plot:   # 2D plot of wavefront 
        temp = wf_fc
        lim_region = False
        plot_region = [-10, 10, -10, 10]

        b = plt.figure(figsize=(10,4.5))
        plt.subplot(121)
        plt.imshow(np.abs(temp).transpose(), origin='lower', 
                   extent=[x_ax.min()/1e3, x_ax.max()/1e3, 
                           y_ax.min()/1e3, y_ax.max()/1e3])
        if lim_region: 
            plt.axis(plot_region)
        plt.title('Amplitude'); plt.colorbar()
        plt.xlabel('X Position [um]'); plt.ylabel('Y Position [um]')
        plt.subplot(122)
        plt.imshow(np.angle(temp).transpose(), origin='lower', 
                   extent=[x_ax.min()/1e3, x_ax.max()/1e3, 
                           y_ax.min()/1e3, y_ax.max()/1e3])
        if lim_region: 
            plt.axis(plot_region)
        plt.title('Phase'); plt.colorbar()
        plt.xlabel('X Position [um]'); plt.ylabel('Y Position [um]')
        plt.tight_layout()
        del temp, plot_region, lim_region
    del wf_fc_amp, wf_fc_ph, x_ax, y_ax, flag_plot
    return wf_fc,a,b