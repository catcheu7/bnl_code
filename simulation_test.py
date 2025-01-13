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