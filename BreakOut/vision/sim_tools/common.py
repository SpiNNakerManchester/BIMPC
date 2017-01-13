from __future__ import print_function
import numpy as np
import pylab as plt
import time
import random
import cv2
from cv2 import cvtColor as convertColor, COLOR_BGR2GRAY, COLOR_GRAY2RGB, \
                resize, imread, imwrite
try:
    from cv2.cv import CV_INTER_NN, \
                       CV_INTER_AREA, \
                       CV_CAP_PROP_FRAME_WIDTH, \
                       CV_CAP_PROP_FRAME_HEIGHT, \
                       CV_CAP_PROP_FPS, \
                       CV_LOAD_IMAGE_GRAYSCALE
except:
    from cv2 import INTER_NEAREST as CV_INTER_NN, \
                    INTER_AREA as CV_INTER_AREA, \
                    CAP_PROP_FRAME_WIDTH as CV_CAP_PROP_FRAME_WIDTH, \
                    CAP_PROP_FRAME_HEIGHT as CV_CAP_PROP_FRAME_HEIGHT, \
                    CAP_PROP_FPS as CV_CAP_PROP_FPS, \
                    IMREAD_GRAYSCALE as CV_LOAD_IMAGE_GRAYSCALE

from scipy.optimize import curve_fit
from scipy import interpolate

from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility import Timer

import copy
import sys
import pickle
import glob
import os

if sys.version_info.major == 2:
  def range(start, stop=None, step=None):
      start = int(start)
      if stop is None:
          return xrange(start)
      elif step is None:
          step = int(stop)
          return xrange(start, stop)
      else:
          stop = int(stop)
          step = int(step)
          return xrange(start, stop, step)

EXC, INH = 0, 1
ROW, COL = 0, 1
OFF, ON = 0, 1
D2R = np.pi/180.
R2D = 180./np.pi

def deg2rad(d):
    return D2R*d

def rad2deg(r):
    return R2D*r

def normalize(mat):
  w = 1./np.sum(np.abs(mat))
  return mat*w, w

def conv2one(mat):
  w = 1./np.sqrt(np.sum(mat**2))
  return mat*w, w

def sum2zero(mat):
  return mat - np.mean(mat)

