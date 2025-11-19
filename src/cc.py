"""
:module: src/cc.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: This script provides DAS Ambient Noise Processing workdflow
"""
import torch
import logging
import numpy as np
from torch import nn
import scipy.signal as signal
from scipy.signal import butter, filtfilt, convolve, detrend
from utils import convert_to_numpy, convert_to_tensor, runtime