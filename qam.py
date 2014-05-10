from numpy import random
from scipy import signal
import numpy as np
import pylab as plt

DECODE = {
    -2:{-2:0, -1:1, 2:2,  1:3},
    -1:{-2:4, -1:5, 2:6,  1:7},
     2:{-2:8, -1:9, 2:10, 1:11},
     1:{-2:12,-1:13,2:14, 1:15},
}

CODE = np.array((
        -2-2j,
        -2-1j,
        -2+2j,
        -2+1j,
        -1-2j,
        -1-1j,
        -1+2j,
        -1+1j,
        +2-2j,
        +2-1j,
        +2+2j,
        +2+1j,
        +1-2j,
        +1-1j,
        +1+2j,
        +1+1j))/2.0

#TODO implement these
def hex_to_symbols():
    return

def symbols_to_hex(symbols):
    decoded = []
    for s in symbols:
        try:
            decoded.append(DECODE[int(s.real)][int(s.imag)])
        except KeyError as e:
            decoded.append(-1)

    return decoded

def mod_QAM16(bits, prefix, f0=1800, tbw=4, fs=48000, baud=300, shaped=True, plot=False):
    Ns = fs/baud
    #code = { 2: -2+2j, 6: -1+2j, 14: 1+2j, 10: 2+2j,
    #        3: -2+1j, 7: -1-1j, 15: 1+1j, 11: 2+1j,
    #        1: -2-1j, 5: -1-1j, 13: 1-1j, 9: 2-1j,
    #        0: -2-2j, 4: -1-2j, 12: 1-2j, 8: 2-2j}


    code = np.array((-2-2j,
        -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j,2+1j,1-2j,+1-1j,1+2j,1+1j))/2

    Nbits = len(bits)
    N = Nbits * Ns

    M = np.tile(CODE[bits],(1,Ns))
    M_prefix = np.tile(CODE[prefix],(1,Ns))
    t = np.r_[0.0:N]/fs
    t_prefix = np.r_[0.0:len(prefix)*Ns]/fs
    if plot:
        fig = plt.figure(figsize = (16,4))
        plt.plot(t_prefix, M_prefix.real.ravel())
        plt.ylim(-3,3)
        plt.title("Real Values of Sent Symbols")

        fig = plt.figure(figsize = (16,4))
        plt.plot(t_prefix, M_prefix.imag.ravel())
        plt.ylim(-3,3)
        plt.title("Imaginary Values of Sent Symbols")

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
    QAM = QAM.real
   #if shaped:
   #    x = np.r_[-2:2:(1.0/147)]
   #    h = np.sinc(x)*signal.hann(147*4)
   #    impulses = np.zeros(len(bits)*Ns)
   #    for i,b in enumerate(bits):
   #        impulses[i*Ns]=code[int(b)]
   #    QAM = signal.fftconvolve(impulses, h)
   #   #print len(QAM)
   #    QAM = QAM[(len(QAM)-N)/4:-(len(QAM)-N)*3/4]
   #   #print len(t)
   #   #print len(QAM)
   #    #t1 = r_[0.0:len(QAM)]/fs

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
    #frange = np.linspace(-fs/2,fs/2,len(r))
    #frange_filt = np.linspace(-fs/2,fs/2,len(r_lp))
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

def detect_sync(r, i, sync, sync_bits, fs=48000, baud=300, plot=False):
    Ns = fs/baud
    corr_r = abs(signal.correlate(r.ravel(), sync.real.ravel(),"same"))
    corr_i = abs(signal.correlate(i.ravel(), sync.imag.ravel(),"same"))
    corr_index = np.argmax(corr_r)-sync_bits/2*Ns

    if plot:
        print np.max(np.abs(corr_r))
        print np.max(np.abs(corr_i))
        fig = plt.figure(figsize = (16,4))
        plt.plot(corr_r)
        plt.plot(corr_i)
        plt.title("Correlation of signal with known prefix")

    return corr_index

def decode_symbols(r, i, corr_index, r0, i0, Nbits, fs=48000, baud=300,
        plot=False):
    Ns = fs/baud
    r = r[corr_index:]
    i = i[corr_index:]

    r0 = r0[corr_index:]
    i0 = i0[corr_index:]

   #print len(r)
   #print len(r0)
   #print len(i0)

    r = r/np.max(r)*2.2 #change this 2 depending on the input amplitude
    i = i/np.max(i)*2.2 #change this 2 depending on the input amplitude

    if plot:
        fig = plt.figure(figsize = (16,4))
        plt.plot(2*r0)
        plt.plot(r)
        plt.title('Real part, raw input and normalized')

        fig = plt.figure(figsize = (16,4))
        plt.plot(2*i0)
        plt.plot(i)
        plt.title('Imaginary part, raw input and normalized')

    ####Decode
    idx = np.r_[Ns/2:len(r):Ns]

    #r = np.around(r)
    #i = np.around(i)

    r_dec = np.around(r[idx])
    i_dec = np.around(i[idx])

    if plot:
        fig = plt.figure(figsize = (16,4))
        plt.plot(2*r0)
        plt.plot(r)
        plt.plot(np.around(r))
        plt.stem(idx, r_dec)
        plt.title('Real part, decoded by sampling values as indicated')

        fig = plt.figure(figsize = (16,4))
        plt.plot(2*i0)
        plt.plot(i)
        plt.plot(np.around(i))
        plt.stem(idx, i_dec)
        plt.title('Imaginary part, decoded by sampling values as indicated')
        fig = plt.figure(figsize = (8,8))
        plt.scatter(r0*2, i0*2, c='r')
        plt.scatter(r_dec, i_dec, c='g')

        plt.title('Constellation of input message vs decoded symbols')

    return r_dec + 1j*i_dec

def test_ber(sigma_range = np.r_[0.01 : 1.478 : 0.1], show_plots=False, verbose=False):
    BER = []
    np.random.seed(seed=1)

    for sigma in sigma_range: #np.r_[1.487 : 1.488 : 0.00001]:#np.r_[0.1 : 8.0 : 0.2]:
        np.random.seed(seed=1)
        rbits = np.int16(random.rand(6,1)*15)
        prefix = np.array([[0],[2],[10],[8], [0],[2],[10],[8]])
        bits = np.int16(random.rand(26,1)*15)

        Nbits = len(rbits) + len(prefix) + len(bits)  # number of bits
        #print "Generating %d random bits"%(Nbits)
        bits = np.array(rbits.tolist() + prefix.tolist() + bits.tolist())

        QAM, M, t, M_prefix, t_prefix = mod_QAM16(bits, prefix, f0=1800, tbw=4, fs=48000, baud=300, shaped=False, plot=False)
        #sigma = 0.6
        QAM += sigma*random.randn(len(QAM))

        i, q = demod_QAM16(QAM, t, f0=1800, fs=48000)

        corr_index = detect_sync(i, q, M_prefix, len(prefix), plot=show_plots)

        symbols = decode_symbols(i, q, corr_index, M.real.ravel(), M.imag.ravel(), Nbits-len(rbits), plot=show_plots)

        #print symbols
        #print bits.tolist()[6:]

        decoded = symbols_to_hex(symbols)
        error_count = 1.0*np.sum([s[0]!=s[1][0] for s in zip(decoded, bits.tolist()[6:])])
        if verbose:
            print 'Error count: %d'%(error_count)
            print 'sigma=%f errors:%f'%(sigma, error_count/Nbits)
        BER.append(error_count/Nbits)

    # plot
    f = plt.figure()
    plt.loglog(1/(sigma_range), BER)
    plt.title("Empirical BER for QAM Demodulation and Correlation")
    plt.xlabel("SNR")
    plt.ylabel("BER")
