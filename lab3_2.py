#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import Queue
import threading,time
import sys

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from numpy import fft
from scipy import signal
from numpy import mean
from numpy import power
from scipy import integrate

import threading,time
import multiprocessing

from rtlsdr import RtlSdr
from  scipy.io.wavfile import read as wavread

from lab3_1 import myspectrogram_hann_ovlp

def plot_sequence(t, sig, bits, fs=44100, modulation="", show=False):
    bit_string = np.array2string(np.transpose(np.uint8(bits)))
    # Plot the time domain signal
    plt.fig = plt.figure(figsize = (16,4))
    plt.plot(t,sig)
    plt.xlabel('time [s]')
    plt.title('%s signal of the sequence: %s'%(modulation, bit_string))

    # Plot the spectrum
    plt.fig = plt.figure(figsize = (16,4))
    plt.plot(np.linspace(-fs/2,fs/2,len(sig)),
            np.abs(fft.fftshift(fft.fft(sig))))
    plt.xlabel('f[hz]')
    plt.title('Spectrum of %s signal of the sequence: %s'%(modulation, bit_string))

    if show:
        plt.show()

def ook(bits, f0, tbw=4, fs=44100, baud=300, Nbits=10, shaped=True, plot=False):
    """
    Using the random bit sequence chosen above, generate a new sequence with either zero or discrete impulses spaced fs/baud samples apart. For example a sequence of 1010 would have an impulse at position 0, an impulse at position 294 and zeros elsewhere
    Generate a TBW = 4 windowed sinc pulse with zero-crossing occuring every 147 samples.
    Convolve the sinc with the impulse train to generate a sinc OOK/ASK signal.
    modulate the result to 1800Hz
    Plot the signal and its spectrum


    M = np.tile(bits,(1,Ns))
    # calculate the TBW of the filter
    """
    Ns = fs/baud
    N = Nbits*Ns

    M = np.tile(bits,(1,Ns))
    t = r_[0.0:N]/fs
    if shaped:
        # Generate a windowed sinc of 147 sample
        x = r_[-2:2:(1.0/147)]
        h = np.sinc(x)*signal.hann(147*4)
        impulses = np.zeros(len(bits)*Ns)
        for i in range(len(bits)):
            if bits[i] ==1:
                impulses[i*Ns]=1
            else:
                impulses[i*Ns]=-1

        plt.figure()
        plt.plot(h)
        if plot:
            plot_sequence(t, impulses, bits, modulation="OOK")

        # Apply the windowed sinc to the original OOK signal
        OOK = signal.fftconvolve(h, impulses)
        OOK = OOK[(len(OOK)-N)/4:]
        t1 = r_[0.0:len(OOK)]/fs
        OOK *= np.sin(2*np.pi*f0*t1)
        if plot:
            plot_sequence(t1, OOK, bits, modulation="OOK")
    else:
        OOK = M.ravel()*np.sin(2*np.pi*f0*t)
        if plot:
            plot_sequence(t, OOK, bits, modulation="OOK")

    return OOK

def bpsk(bits, f0, fs=44100, baud=300, Nbits=10, shaped=True, plot=False):
    """
    np.random.seed(seed=1)
    bits = randn(Nbits,1) > 0
    M = np.tile(bits*2-1,(1,Ns))
    t = r_[0.0:N]/fs
    BPSK = M.ravel()*sin(2*pi*f0*t)

    fig = figure(figsize = (16,4))
    plot(t,BPSK)
    xlabel('time [s]')
    title('BPSK signal of the sequence:'+ np.array2string(transpose(np.uint8(bits))))
    """
    Ns = fs/baud
    N = Nbits*Ns

    M = np.tile(bits*2-1,(1,Ns))
    t = r_[0.0:N]/fs

    if shaped:
        # Generate a windowed sinc of 147 sample
        x = r_[-2:2:(1.0/147)]
        h = np.sinc(x)*signal.hann(147*4)
        impulses = np.zeros(len(bits)*Ns)
        for i in range(len(bits)):
            if bits[i] ==1:
                impulses[i*Ns]=1
            else:
                impulses[i*Ns]=-1
        # Apply the windowed sinc to the original OOK signal
        BPSK = signal.fftconvolve(h, impulses)
        BPSK = BPSK[(len(BPSK)-N)/4:]
        t1 = r_[0.0:len(BPSK)]/fs
        BPSK *= np.sin(2*np.pi*f0*t1)
        if plot:
            plot_sequence(t1, BPSK, bits, modulation="BPSK")
    else:
        BPSK = M.ravel()*sin(2*pi*f0*t)
        if plot:
            plot_sequence(t, BPSK, bits, modulation="BPSK")

    return BPSK

def qam(bits, f0, tbw=4, fs=44100, baud=300, Nbits=16, shaped=True, plot=False):
    """
    #code = { 2: -2+2j, 6: -1+2j, 14: 1+2j, 10: 2+2j,
    #        3: -2+1j, 7: -1-1j, 15: 1+1j, 11: 2+1j,
    #        1: -2-1j, 5: -1-1j, 13: 1-1j, 9: 2-1j,
    #        0: -2-2j, 4: -1-2j, 12: 1-2j, 8: 2-2j}
    """
    Ns = fs/baud
    Nbits = 16  # number of bits
    N = Nbits * Ns
    code = np.array((-2-2j, -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j+2+1j,1-2j,+1-1j,1+2j,1+1j))/2
    np.random.seed(seed=1)
    bits = np.int16(np.random.rand(Nbits,1)*16)
    M = np.tile(code[bits],(1,Ns))
    t = r_[0.0:N]/fs

    if shaped:
        # Generate a windowed sinc of 147 sample
        x = r_[-2:2:(1.0/147)]
        h = np.sinc(x)*signal.hann(147*4)
        impulses = np.zeros(len(bits)*Ns)
        for i,b in enumerate(bits):
            impulses[i*Ns]=code[int(b)]

        # Apply the windowed sinc to the original OOK signal
        QAM = signal.fftconvolve(h, impulses)
        QAM = QAM[(len(QAM)-N)/4:]
        t1 = r_[0.0:len(QAM)]/fs
        QAM = QAM*np.exp(1j*2*np.pi*f0*t1)/np.sqrt(2)/2
        if plot:
            plot_sequence(t1, QAM.real, bits, modulation="QAM=16")
    else:
        # these are two ways of producing the same result:
        #QAM = (M.real.ravel()*cos(2*pi*f0*t) - M.imag.ravel()*sin(2*pi*f0*t))/2/sqrt(2)
        QAM = np.real(M.ravel()*np.exp(1j*2*np.pi*f0*t))/np.sqrt(2)/2
        if plot:
            plot_sequence(t, QAM.real, bits, modulation="QAM=16")

    return QAM

def bfsk(bits, f0, delta_f=600, fs=44100, baud=300, Nbits=10, shaped=True, plot=False):
    """
    N = Nbits * Ns
    M = np.tile(bits*2-1,(1,Ns))

    # compute phase by integrating frequency
    ph = 2*np.pi*np.cumsum(f0 + M.ravel()*delta_f)/fs
    t = r_[0.0:N]/fs
    BFSK = sin(ph)
    print len(BFSK)
    if plot:
        plot_sequence(t, BFSK, bits, modulation="BFSK")
    return BFSK
    """
    Ns = baud/fs
    np.random.seed(seed=1)
    Nbits = 10
    N = Nbits * Ns
    bits = np.random.randn(Nbits,1) > 0
    print len(bits)
    M = np.tile(bits*2-1,(1,Ns))
    print np.shape(M)
    print M
    delta_f = 600

    # compute phase by integrating frequency
    ph = 2*np.pi*np.cumsum(f0 + M.ravel()*delta_f)/fs
    print len(ph)
    t = r_[0.0:N]/fs
    FSK = np.sin(ph)
    print len(FSK)

    plt.fig = plt.figure(figsize = (16,4))
    plt.plot(t,FSK)
    plt.xlabel('time [s]')
    plt.title('FSK signal of the sequence:'+
    np.array2string(np.transpose(np.uint8(bits))))

def mfsk():
    """
    np.random.seed(seed=1)
    Nbits = 10
    N = Nbits * Ns
    bits = randn(Nbits,1) > 0
    M = np.tile(bits*2-1,(1,Ns))
    delta_f = 600

    # compute phase by integrating frequency
    ph = 2*pi*cumsum(f0 + M.ravel()*delta_f)/fs
    t = r_[0.0:N]/fs
    FSK = sin(ph)

    fig = figure(figsize = (16,4))
    plot(t,FSK)
    xlabel('time [s]')
    title('FSK signal of the sequence:'+ np.array2string(transpose(np.uint8(bits))))
    """

def main():
    np.random.seed(seed=1)
    Nbits=10
    rand_bits = np.random.randn(Nbits,1) > 0
    #ook(rand_bits, 1800, shaped=False, plot=True)
    ook(rand_bits, 1800, shaped=True, plot=True)
    #bpsk(rand_bits, 1800, shaped=True, plot=True)
    #bpsk(rand_bits, 1800, shaped=False, plot=True)

    #bfsk(rand_bits, 1800, shaped=True, plot=True)
    #bpsk(rand_bits, 1800, shaped=False, plot=True)

    #qam(rand_bits, 1800, shaped=False, plot=True)
    #qam(rand_bits, 1800, shaped=True, plot=True)

    plt.show()

if __name__ == "__main__":
    main()
