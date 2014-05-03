import numpy as np
import matplotlib.pyplot as plt
import Queue
import bitarray
import ax25

from numpy import r_
from numpy import fft
from numpy import random
from scipy import signal
from  scipy.io.wavfile import read as wavread

'''
def afsk1200_1(bits, fs=48000):
    Args:
    bits (bitarray.bitarray)
    #the function will take a bitarray of bits and will output an AFSK1200 modulated signal of them, sampled at 44100Hz
    baud = 1200
    fc = 1700
    delta_f = 500.0
    fs = fs*4
    Ns = fs/baud
    #print Ns
    decoded = []
    for bit in list(bits):
        if bit == 1:
            for i in range(0, int(Ns)):
                decoded.append(1.0)
        if bit == 0:
            for i in range(0, int(Ns)):
                decoded.append(-1.0)

    t = r_[0:len(decoded)]/(fs*1.0)

    sig = cos(2*pi*fc*t + 2*pi*delta_f*cumsum(decoded)/fs)
    afsk = sig[::4]

    if plot:
        fig, axarr = plt.subplots(2, 1, figsize=(15,12))

        axarr[0].plot(afsk)
        axarr[0].set_title('AFSK 1200 Signal')
        axarr[0].set_xlabel('t (s)')

        axarr[1].plot(np.linspace(0,fs/4,len(afsk)/4),
                 np.abs(fft.fft(afsk))[:len(afsk)/4])
        axarr[1].set_title('AFSK 1200 Spectrum')
        axarr[1].set_xlabel('f (MHz)')

        fig.tight_layout()

    return sig
'''

def afsk1200(bits, fs=48000.0):
    #TODO: docstring
    #the function will take a bitarray of bits and will output an AFSK1200 modulated signal of them, sampled at 44100Hz
    fc = 1700
    delta_f = 500.0
    fs = 44100*4
    baud = 1200
    Ns = fs/baud

    code = {0:[-1.0 for k in range(Ns)], 1:[1.0 for k in range(Ns)]}
    to_mod = []
    for bit in list(bits):
        to_mod += code[bit]

    t = r_[0:len(to_mod)]/(fs*1.0)
    afsk = np.cos(2*np.pi*fc*t + 2*np.pi*delta_f*np.cumsum(to_mod)/fs)
    afsk = afsk[::4]

    #TODO: plotting code

    return afsk

def nc_afskDemod(sig, tbw=2.0, fs=48000.0):
    #TODO: docstring
    #  non-coherent demodulation of afsk1200
    # function returns the NRZI (without rectifying it)
    baud = 1200.0
    M = int(2/(1200.0/fs))
    h = signal.firwin(M, 600, nyq=fs/2)
    t = r_[0:len(h)]/fs

    bp_1200 = h*np.exp(t*1j*1200*2.0*np.pi)
    bp_2200 = h*np.exp(t*1j*2200*2.0*np.pi)

    filt_1200 = abs(signal.fftconvolve(bp_1200, sig))
    filt_2200 = abs(signal.fftconvolve(bp_2200, sig))
    NRZI = filt_2200 - filt_1200

    return NRZI

def fm_afskDemod(sig, TBW=4, N=74, fs=48000.0):
    #TODO: add docstring
    #  non-coherent demodulation of afsk1200
    # function returns the NRZI (without rectifying it)
    baud = 1200.0
    bandwidth = 2*500.0 + baud

    #TODO fix this
    M = int(2/(bandwidth/fs))
    h = signal.firwin(74.0, bandwidth, nyq=fs/2)

    t = r_[0.0:len(h)]/fs
    fc = 1700.0
    h = h*np.exp(1j*2*np.pi*fc*t)
    output = signal.fftconvolve(h, sig)

    temp = output*np.conjugate(np.roll(output, 1))
    NRZ_fm = np.angle(temp)/3

    h2 = signal.firwin(74.0, 1200, nyq=fs/2)
    NRZ_fm = signal.fftconvolve(NRZ_fm, h2)

    NRZ_fm = (NRZ_fm*fs/(2.0*np.pi)-550.0)/500.0

    return NRZ_fm

def decode_bits(NRZ, Nbits, offset, fs=48000.0):
    baud = 1200.0
    bit_period = fs/baud
    idx = []
    i = offset
    for x in range(Nbits):
        idx.append(int(i))
        i = i + bit_period

    idx = np.array(idx)
    bits_dec = bitarray.bitarray((NRZ[idx]>0).tolist())
    return bits_dec

def test_ber():
    # Generate a random bit steam
    Nbits = 1000
    bits=bitarray.bitarray((random.rand(Nbits)>0.5).tolist())

    # modulate
    sig = afsk1200(bits,fs=44100.0)


    # add noise
    sig_n = sig + 1*random.randn(len(sig))

    # with noise or without?
    # demodulate and decode bits with non-coherent and FM demodulators
    NRZa_nc = nc_afskDemod(sig_n, tbw=2.0)
    NRZa_fm = fm_afskDemod(sig_n, TBW=4)

    # Compute Error Bit Rate Curves
    BER_nc = []
    BER_fm = []

    for sigma in r_[0.1:8.0:0.2]:
        bits_temp=bitarray.bitarray((random.rand(Nbits)>0.5).tolist())

        # modulate and add noise
        sig = afsk1200(bits_temp,fs=44100.0)
        sig_n = sig + sigma*random.randn(len(sig))

        # demodulate and decode bits with non-coherent and FM demodulators
        NRZa_nc = nc_afskDemod(sig_n, tbw=2.0, fs=44100.0)
        NRZa_fm = fm_afskDemod(sig_n, TBW=4, N=74)
        NRZ_nc = np.sign(NRZa_nc)
        NRZ_fm = np.sign(NRZa_fm)

        E_nc = 0
        E_fm = 0

        fs = 44100.0

        bits_dec_nc = decode_bits(NRZ_nc, Nbits, 56, fs=fs)
        E_nc = int(bitarray.bitdiff(bits_temp, bits_dec_nc[0: min(len(bits_temp), len(bits_dec_nc)) ]))

        bits_dec_fm = decode_bits(NRZ_fm, Nbits, 92, fs=fs)
        E_fm = int(bitarray.bitdiff(bits_temp, bits_dec_fm[0:
            min(len(bits_temp), len(bits_dec_fm)) ]))

        BER_nc.append(1.0*E_nc/len(bits_temp))
        BER_fm.append(1.0*E_fm/len(bits_temp))

    print BER_nc
    print BER_fm

    # plot
    f = plt.figure()
    plt.loglog(1/(r_[0.1:8.1:0.2]),BER_nc)
    plt.loglog(1/(r_[0.1:8.1:0.2]),BER_fm,'r')
    plt.title("empirical BER for AFSK demodulation")
    plt.xlabel("SNR")
    plt.ylabel("BER")
    plt.legend(("non-coherent","FM"))
    plt.show()

def NRZ2NRZI(NRZ):
    NRZI = NRZ.copy()
    current = True
    for n in range(0,len(NRZ)):
        if NRZ[n] :
            NRZI[n] = current
        else:
            NRZI[n] = not(current)
        current = NRZI[n]
    return NRZI

def genPTT(plen=150, zlen=400, fs=44100.0, plot=False):
    """Function generates a short pulse to activate the VOX

    Args:
        plen (int): 2000Hz pulse len in ms
        zlen (int): total length of the signal in ms (zero-padded)
        fs (float):  sampling frequency in Hz
        plot (boolean, optional): plot the time domain output and its spectrogram (default=false)

    Returns:
        numpy.array: the ptt signal, sinusoid pulse at 2000Hz
    """
    print plen, zlen, fs

    t = r_[0.0:(plen/1000.0)*fs]/fs
    pulse = 0.5*np.sin(2*np.pi*t*2000)
    ptt = np.append(pulse, np.zeros(int((zlen-plen)/1000*fs)))

    if plot:
        plt.figure(figsize=(15,6))
        plt.plot(np.linspace(0,zlen, len(ptt)), ptt)
        plt.title('PTT Signal')
        plt.xlabel('Time (ms)')

        myspectrogram_hann_ovlp(ptt, 128.0, fs, 0.0)
        plt.title('PTT Spectrogram')
        plt.show()
    return ptt

def gen_packet(position=False, source='KK6KKT', dest = "APDSP", info='', digi=b'WIDE1-1,WIDE2-1', verbose=False):
    # Uncomment to Send Email
    info = ":EMAIL    :viyer@berkeley.edu EE123 Lab 3"
    if position:
        info = "=3752.50N/12215.43WKThis is Cory Hall!"

    #Uncomment to show yourself on top of Mt  everest
    #info = "=2759.16N/08655.30E[I'm on the top of the world"

    #uncomment to send to everyone on the APRS system near you
    #info = ":ALL      : CQCQCQ I would like to talk to you!"



    # uncomment to send a status message
    # info = ">I like radios"

    packet = ax25.UI(
            destination=dest,
            source=source,
            info=info,
            digipeaters=digi.split(b','),
            )

    if verbose:
        print(packet.unparse())

    return packet

def send_aprs(position=False, verbose=False, plot=False):
    packet = gen_packet(position)
    msg = afsk1200(NRZ2NRZI(bitarray.bitarray(np.zeros(160).tolist())+packet.unparse()))
    if plot:
        fs = 48000
        fc = 1700

    try:
        ptt =  genPTT(150, 1000, 48000.0, plot=False)
        print 'Sending PTT...'
       #transmit_and_capture(ptt, 'test', 4, '', verbose, plot=False)
        p = pyaudio.PyAudio()   # Create a PyAudio object

        # Get device numbers
        din, dout, dusb = audio_dev_numbers(p, in_name='USB', out_name='default', debug=verbose)

       #play_audio(ptt/2.0, p, 48000.0, dusb)
       #play_audio(ptt/2.0, p, 48000.0, dusb)
       #time.sleep(1)
        to_send = np.append(ptt, msg)
        to_send = np.append(to_send, msg)
        play_audio(to_send, p, 48000.0, dusb)
    except IOError:
        type, value, traceback = sys.exc_info()
        print('Error loading %s: %s' % (value.filename, value.strerror))
    except Exception as e:
        print 'Error: '+str(e)
        print e
    finally:
        print 'Cleaning up...'
        #sdr.close()
        p.terminate()
        print 'Closed SDR and PyAudio'

# function to generate a checksum for validating packets
def genfcs(bits):
    # Generates a checksum from packet bits
    fcs = ax25.FCS()
    for bit in bits:
        fcs.update_bit(bit)

    digest = bitarray.bitarray(endian="little")
    digest.frombytes(fcs.digest())

    return digest

# function to parse packet bits to information
def decodeAX25(bits):
    ax = ax25.AX25()
    ax.info = "bad packet"


    bitsu = ax25.bit_unstuff(bits[8:-8])

    if (genfcs(bitsu[:-16]).tobytes() == bitsu[-16:].tobytes()) == False:
        #print("failed fcs")
        return ax

    bytes = bitsu.tobytes()
    ax.destination = ax.callsign_decode(bitsu[:56])
    source = ax.callsign_decode(bitsu[56:112])
    if source[-1].isdigit() and source[-1]!="0":
        ax.source = b"".join((source[:-1],'-',source[-1]))
    else:
        ax.source = source[:-1]

    digilen=0

    if bytes[14]=='\x03' and bytes[15]=='\xf0':
        digilen = 0
    else:
        for n in range(14,len(bytes)-1):
            if ord(bytes[n]) & 1:
                digilen = (n-14)+1
                break

    ax.digipeaters =  ax.callsign_decode(bitsu[112:112+digilen*8])
    ax.info = bitsu[112+digilen*8+16:-16].tobytes()

    return ax

def detectFrames(NRZI):
    # function looks for packets in an NRZI sequence and validates their checksum

    # compute finite differences of the digital NRZI to detect zero-crossings
    dNRZI = NRZI[1:] - NRZI[:-1]
    # find the position of the non-zero components. These are the indexes of the zero-crossings.
    transit = np.nonzero(dNRZI)[0]
    # Transition time is the difference between zero-crossings
    transTime = transit[1:]-transit[:-1]

    # loop over transitions, convert to bit streams and extract packets
    dict = { 1:bitarray.bitarray([0]), 2:bitarray.bitarray([1,0]), 3:bitarray.bitarray([1,1,0]),
            4:bitarray.bitarray([1,1,1,0]),5:bitarray.bitarray([1,1,1,1,0]),6:bitarray.bitarray([1,1,1,1,1,0])
            ,7:bitarray.bitarray([1,1,1,1,1,1,0])}

    state = 0; # no flag detected yet

    packets =[]
    tmppkt = bitarray.bitarray([0])
    lastFlag = 0  # position of the last flag found.

    for n in range(0,len(transTime)):
        Nb = round(transTime[n]/36.75)  # maps intervals to bits. Assume 44100Hz and 1200baud
        if (Nb == 7 and state ==0):
            # detected flag frame, start collecting a packet
            tmppkt = tmppkt +  dict[7]
            state = 1  # packet detected
            lastFlag = transit[n-1]
            continue
        if (Nb == 7 and state == 1):
            # detected end frame successfully
            tmppkt = tmppkt + dict[7]

            # validate checksum
            bitsu = ax25.bit_unstuff(tmppkt[8:-8]) # unstuff bits
            if (genfcs(bitsu[:-16]).tobytes() == bitsu[-16:].tobytes()) :
                # valid packet
                packets.append(tmppkt)
            tmppkt  = bitarray.bitarray([0])
            state = 0
            continue

        if (state == 1 and Nb < 7 and Nb > 0):
            # valid bits
            tmppkt = tmppkt + dict[Nb]
            continue
        else:
            # not valid bits reset
            state = 0
            tmppkt  = bitarray.bitarray([0])
            continue

    if state == 0:
        lastFlag = -1

    # if the state is 1, which means that we detected a packet, but the buffer ended, then
    # we return the position of the beginning of the flag within the buffer to let the caller
    # know that there's a packet that overlapps between two buffer frames.

    return packets, lastFlag

def example():
    sig = wavread("ISSpkt.wav")[1]
    NRZIa = nc_afskDemod(sig)
    fig = plt.figure(figsize=(16,4))
    plt.plot(NRZIa)
    NRZI = np.sign(NRZIa)
    packets ,lastflag = detectFrames(NRZI)
    ax = decodeAX25(packets[0])
    print("Dest: %s | Source: %s | Digis: %s | %s |" %(ax.destination ,ax.source ,ax.digipeaters,ax.info))
    print lastflag

def test_decoding():
    # Load ISS Packet
    Qin = Queue.Queue()
    sig = wavread("ISSpkt_full.wav")[1]
    print len(sig)
    for n in r_[0:len(sig):1024]:
        Qin.put(sig[n:n+1024])
    Qin.put("END")

    length = 43
    end = False
    count = 1
    while(Qin.not_empty):
        buf = np.array([])
        for i in range(length):
            chunk = Qin.get()
            if chunk == "END":
                print chunk
                end = True
                break
            else:
                buf = np.append(buf, chunk)
        NRZIa = nc_afskDemod(buf)
        NRZI = np.sign(NRZIa)
        packets, lastflag = detectFrames(NRZI)
        # make recursive?
        while(lastflag > 0):
            for i in range(20):
                chunk = Qin.get()
                if chunk == "END":
                    print chunk
                    end = True
                    break
                else:
                    buf = np.append(buf, chunk)
            NRZIa = nc_afskDemod(buf)
            NRZI = np.sign(NRZIa)
            packets, lastflag = detectFrames(NRZI)
            if lastflag>0:
                print lastflag

        for p in packets:
            #print "%d. %s"%(count, str(decodeAX25(p)))
            ax = decodeAX25(p)
            print ("%d. Dest: %s | Source: %s | Digis: %s | %s" %(count, ax.destination ,ax.source , ax.digipeaters, ax.info))
            count += 1
        if end:
            return
            #ax = decodeAX25(p)
          # print ("Dest: %s | Source: %s | Digis: %s | %s" %(ax.destination ,ax.source , ax.digipeaters, ax.info))
        #print len(packets)
        #print len(buf)
        #print lastflag

    #load(Qin)
    #test(350)
   #test(1000)
   #test(1000)
   #test(1000)
   #test(1000)
   #test(1000)
   #test(1000)
   #test(1000)
   #test(1000)

if __name__ == "__main__":
    #test_ber()
    #example()
     test_decoding()
