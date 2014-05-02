# Import functions and libraries
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
fs = 44100  # sampling rate
baud = 300  # symbol rate
Ns = fs/baud
f0 = 1800
prefix = [[0],[2],[10],[8]]

#code = { 2: -2+2j, 6: -1+2j, 14: 1+2j, 10: 2+2j,
#        3: -2+1j, 7: -1-1j, 15: 1+1j, 11: 2+1j,
#        1: -2-1j, 5: -1-1j, 13: 1-1j, 9: 2-1j,
#        0: -2-2j, 4: -1-2j, 12: 1-2j, 8: 2-2j}
Nbits = 16  + len(prefix)# number of bits
print "nbits: ",Nbits
N = Nbits * Ns
code = np.array((-2-2j, -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j+2+1j,1-2j,+1-1j,1+2j,1+1j))/2
np.random.seed(seed=1)
bits = np.append(prefix,np.int16(rand(Nbits,1)*16) )
print bits
M = np.tile(code[bits],(1,Ns))
t = r_[0.0:N]/fs


#fig = figure()
#scatter(M.real.ravel(), M.imag.ravel())

#xlabel('Real')
#ylabel('Imag')
#title("QAM=16 of the sequence:"+ np.array2string(transpose(bits)))

#fig = figure()
#plot(M.real.ravel())
#figure()
#plot(M.imag.ravel())


# these are two ways of producing the same result:
#QAM = (M.real.ravel()*cos(2*pi*f0*t) - M.imag.ravel()*sin(2*pi*f0*t))/2/sqrt(2)
QAM = np.real(M.ravel()*exp(1j*2*pi*f0*t))/sqrt(2)/2
#fig = figure(figsize = (16,4))
#plot(t,QAM.real)
#xlabel('time [s]')
#title("QAM=16 of the sequence:"+ np.array2string(transpose(bits)))

QAM = (M.real.ravel()*cos(2*pi*f0*t) - M.imag.ravel()*sin(2*pi*f0*t))/2/sqrt(2)
#QAM = qam(bits, 1800, shaped=True, plot=True)[:len(M.ravel())]
fig = figure(figsize = (16,4))
plot(t,QAM.real)
plot(t,QAM.imag)
xlabel('time [s]')
title("QAM=16 of the sequence:"+ np.array2string(transpose(bits)))
show()
#signal.firwin()
r = QAM*cos(2*pi*f0*t)
i = -QAM*sin(2*pi*f0*t)

#plot(r+i)
lp = signal.firwin(100,pi*f0/4,nyq=44100.0/2.0)
r_lp = signal.fftconvolve(lp,r)
i_lp = signal.fftconvolve(lp, i)

#fig = figure(figsize = (16,4))
frange = np.linspace(-fs/2,fs/2,len(r))
frange_filt = np.linspace(-fs/2,fs/2,len(r_lp))
#plt.plot(frange_filt, abs(fft.fftshift(fft.fft(lp))))

#fig = figure(figsize = (16,4))
#plt.plot(frange, abs(np.fft.fftshift(np.fft.fft(r))))
#plt.plot(frange_filt, abs(np.fft.fftshift(np.fft.fft(r_lp))))

#fig = figure(figsize = (16,4))
#plt.plot(frange, abs(np.fft.fftshift(np.fft.fft(i))))
#plt.plot(frange_filt, abs(np.fft.fftshift(np.fft.fft(i_lp))))

#r_env = abs(r_lp)
#i_env = abs(i_lp)
#fig = figure(figsize = (16,4))
#plot(M.real.ravel()/max(M.real.ravel()))
#plot(r_lp[50:]/max(r_lp))
#title('Actual and filtered real')

#fig = figure(figsize = (16,4))
#plot(M.imag.ravel()/max(M.imag.ravel()))
#plot(i_lp[50:]/max(i_lp))
#title('Actual and filtered imag')

r = r_lp[50:]
i = i_lp[50:]
idx = r_[0:len(r):len(r)/16]

r = r/max(r)*max(M.real.ravel())*2
i = i/max(i)*max(M.imag.ravel())*2

#print "i: ", i



r = np.around(r)
i = np.around(i)



r_dec = np.around(r[idx])
i_dec = np.around(i[idx])




#offset = len(r)/16
offset = len(M.real.ravel())/16
# fig = figure(figsize = (16,4))
# plot(M.real.ravel()*2)
# plot(r)
# stem(r_[0:len(r):offset], r_dec)
# title('Real averaging')
print "len(r)", len(r)
print "len(r_dec)", len(r_dec)
#print "len(M.real.ravel()*4), ", len(M.real.ravel())
r_avg = []
i_avg = []
periods = r_[0:len(r):offset]


for prd in periods[0:-1]:
    current_avg_r = np.average(r[prd:prd+offset])
    r_avg = np.append(r_avg, current_avg_r)
    
    current_avg_i = np.average(i[prd:prd+offset])
    i_avg = np.append(i_avg, current_avg_i)
    
# fig = figure()
# plot(i)

# fig = figure(figsize = (16,4))
# plot(M.imag.ravel()*2)
# plot(i)
# print "i ", i
# stem(r_[0:len(r):offset], i_dec)
# #plot(i_dec)
# title('Imaginary averaging')
# fig = figure(figsize = (8,8))
# scatter(M.real.ravel()*2, M.imag.ravel()*2, c='r')
# scatter(r_dec, i_dec, c='g')

# fig = figure(figsize = (8,8))
# scatter(M.real.ravel()*2, M.imag.ravel()*2, c='r')
# scatter(np.around(r_avg), np.around(i_avg))
# show()
decoded = np.around(r_avg) + 1j*np.around(i_avg) 

finalbits = np.array([])
decoded = decoded/2
#for symb in decoded:
    #print "Original: %s Decoded: %s"%(code[bits][k],decoded[k])
 #   print np.where(code == symb)
 #   finalbits = np.append(finalbits,DECODE[symb])
    
#print "bits: %s "%str(code[bits])
#print "decoded: %s "%str(decoded/2)
#print code[bits]
#print finalbits




