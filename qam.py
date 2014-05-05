from scipy import signal
import numpy as np
import pylab as plt

ENCODE = {
    0: -2-2j,
    1: -2-1j,
    2: -2+2j,
    3: -2+1j,
    4: -1-2j,
    5: -1-1j,
    6: -1+2j,
    7: -1-1j,
    8:  2-2j,
    9:  2-1j,
    10: 2+2j,
    11: 2+1j,
    12: 1-2j,
    13: 1-1j,
    14: 1+2j,
    15: 1+1j,
}

DECODE = {
    -2-2j:0,
    -2-1j:1,
    -2+2j:2,
    -2+1j:3,
    -1-2j:4,
    -1-1j:5,
    -1+2j:6,
    -1-1j:7,
     2-2j:8,
     2-1j:9,
     2+2j:10,
     2+1j:11,
     1-2j:12,
     1-1j:13,
     1+2j:14,
     1+1j:15,
}

#TODO implement these
def hex_to_symbols():
    return

def symbols_to_hex():
    return

def mod_QAM16(bits, prefix, f0=1800, tbw=4, fs=48000, baud=300, shaped=True, plot=False):
    Ns = fs/baud
    #code = { 2: -2+2j, 6: -1+2j, 14: 1+2j, 10: 2+2j,
    #        3: -2+1j, 7: -1-1j, 15: 1+1j, 11: 2+1j,
    #        1: -2-1j, 5: -1-1j, 13: 1-1j, 9: 2-1j,
    #        0: -2-2j, 4: -1-2j, 12: 1-2j, 8: 2-2j}


    code = np.array((-2-2j,
        -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j+2+1j,1-2j,+1-1j,1+2j,1+1j))/2

    Nbits = len(bits)
    N = Nbits * Ns

    M = np.tile(code[bits],(1,Ns))
    M_prefix = np.tile(code[prefix],(1,Ns))
    t = np.r_[0.0:N]/fs
    t_prefix = np.r_[0.0:len(prefix)*Ns]/fs

    fig = plt.figure(figsize = (16,4))
    plt.plot(t_prefix, M_prefix.real.ravel())
    plt.ylim(-3,3)

    fig = plt.figure(figsize = (16,4))
    plt.plot(t_prefix, M_prefix.imag.ravel())
    plt.ylim(-3,3)

    fig = plt.figure()
    plt.scatter(M.real.ravel(), M.imag.ravel())

    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title("QAM=16 of the sequence:"+ np.array2string(np.transpose(bits)))

    #fig = figure(figsize = (16,4))
    #plot(t,QAM.real)
    #xlabel('time [s]')
    #title("QAM=16 of the sequence:"+ np.array2string(transpose(bits)))

    QAM = (M.real.ravel()*np.cos(2*np.pi*f0*t) -
            M.imag.ravel()*np.sin(2*np.pi*f0*t))/2/np.sqrt(2)

    return QAM, M, t, M_prefix, t_prefix

def demod_QAM16(QAM, t, f0=1800, fs=48000):
    r = QAM*np.cos(2*np.pi*f0*t)
    i = -QAM*np.sin(2*np.pi*f0*t)

    #plot(r+i)
    num_taps = 100
    lp = signal.firwin(num_taps, np.pi*f0/4,nyq=fs/2.0)
    r_lp = signal.fftconvolve(lp,r)
    i_lp = signal.fftconvolve(lp, i)

    #fig = figure(figsize = (16,4))
    frange = np.linspace(-fs/2,fs/2,len(r))
    frange_filt = np.linspace(-fs/2,fs/2,len(r_lp))
    #plt.plot(frange_filt, abs(fft.fftshift(fft.fft(lp))))
    '''
    ylim(-3,3)

    fig = figure(figsize = (16,4))
    plt.plot(frange, abs(fft.fftshift(fft.fft(i))))
    plt.plot(frange_filt, abs(fft.fftshift(fft.fft(i_lp))))

    #r_env = abs(r_lp)
    #i_env = abs(i_lp)
    '''
    r_lp = r_lp[num_taps/2:-num_taps/2+1]
    i_lp = i_lp[num_taps/2:-num_taps/2+1]
    return r_lp, i_lp

def detect_sync(r, i, sync, sync_bits, fs=48000, baud=300):
    Ns = fs/baud
    corr_r = abs(signal.correlate(r.ravel(), sync.real.ravel(),"same"))
    corr_i = abs(signal.correlate(i.ravel(), sync.imag.ravel(),"same"))
    corr_index = np.argmax(corr_r)-sync_bits/2*Ns
    print np.max(np.abs(corr_r))
    print np.max(np.abs(corr_i))

    fig = plt.figure(figsize = (16,4))
    plt.plot(corr_r)
    plt.plot(corr_i)
    return corr_index

def decode_symbols(r, i, corr_index, r0, i0, Nbits, fs=48000, baud=300):
    Ns = fs/baud
    r = r[corr_index:]
    i = i[corr_index:]

    r0 = r0[corr_index:]
    i0 = i0[corr_index:]

    print len(r)
    print len(r0)
    print len(i0)

    r = r/np.max(r)*2.2 #change this 2 depending on the input amplitude
    i = i/np.max(i)*2.2 #change this 2 depending on the input amplitude

    fig = plt.figure(figsize = (16,4))
    plt.plot(r0)
    plt.plot(r)
    plt.title('Actual and filtered real')

    fig = plt.figure(figsize = (16,4))
    plt.plot(i0)
    plt.plot(i)
    plt.title('Actual and filtered imag')

    ####Decode
    idx = np.r_[Ns/2:len(r):Ns]

    #r = np.around(r)
    #i = np.around(i)

    r_dec = np.around(r[idx])
    i_dec = np.around(i[idx])

    fig = plt.figure(figsize = (16,4))
    plt.plot(r0)
    plt.plot(r)
    plt.plot(np.around(r))
    plt.stem(idx, r_dec)

    fig = plt.figure(figsize = (16,4))
    plt.plot(i0)
    plt.plot(i)
    plt.plot(np.around(i))
    plt.stem(idx, i_dec)

    fig = plt.figure(figsize = (8,8))
    plt.scatter(r0*2, i0*2, c='r')
    plt.scatter(r_dec, i_dec, c='g')

