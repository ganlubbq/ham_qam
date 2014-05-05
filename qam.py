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

def bits_to_symbols():
    return

def symbols_to_bits():
    return

#def mod_QAM16(bits, f0, tbw=4, fs=44100, baud=300, Nbits=16, shaped=True, plot=False):
    def mod_QAM16(shaped)
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
        QAM = signal.fftconvolve(h, impulses, 'same')
        #QAM = QAM[(len(QAM)-N)/4:]
        #t1 = r_[0.0:len(QAM)]/fs
        QAM = (M.real.ravel()*cos(2*pi*f0*t) - M.imag.ravel()*sin(2*pi*f0*t))/2/sqrt(2)
        if plot:
            plot_sequence(t1, QAM.real, bits, modulation="QAM=16")
    else:
        # these are two ways of producing the same result:
        QAM = (M.real.ravel()*cos(2*pi*f0*t) - M.imag.ravel()*sin(2*pi*f0*t))/2/sqrt(2)
        if plot:
            plot_sequence(t, QAM.real, bits, modulation="QAM=16")

    return QAM

def detect_sync():
    return

def demod_QAM16():
    return

def decode_symbols():
    return
