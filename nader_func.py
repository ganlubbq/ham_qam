import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pyaudio
import Queue
import threading,time
import sys

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from numpy import transpose
from numpy import sinc
from numpy import *
from numpy.random import rand
from numpy.random import randn
from scipy import signal
from scipy import integrate

import threading,time
import multiprocessing

#from rtlsdr import RtlSdr
from numpy import mean
from numpy import power
from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifftshift
#import bitarray
from  scipy.io.wavfile import read as wavread




from qam import DECODE


def randomBits(Nbits):
	np.random.seed(seed=1)
	#bits = np.append(prefix,np.int16(rand(Nbits,1)*Nbits) )
	bits = np.int16(rand(Nbits,1)*Nbits)
	return bits

def makeSymbols(bits, fs, code, Ns):
	#returns M, t


	M = np.tile(code[bits],(1,Ns))
	t = r_[0.0:len(bits)*Ns]/fs
	return M, t

def demod(QAM, f0, t):
	#returns r,i
	r = QAM*cos(2*pi*f0*t)
	i = -QAM*sin(2*pi*f0*t)
	return r,i
