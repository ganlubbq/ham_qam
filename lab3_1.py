#!/usr/bin/env python
# Import functions and libraries
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import Queue
import threading,time
import sys
import cmath

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import integrate

import threading,time
import multiprocessing
import argparse

from rtlsdr import RtlSdr

# function to compute average power spectrum
def avgPS( x, N=256, fs=1):
    M = floor(len(x)/N)
    x_ = reshape(x[:M*N],(M,N)) * np.hamming(N)[None,:]
    X = np.fft.fftshift(np.fft.fft(x_,axis=1),axes=1)
    return r_[-N/2.0:N/2.0]/N*fs, mean(abs(X**2),axis=0)


# Plot an image of the spectrogram y, with the axis labeled with time tl,
# and frequency fl
#
# t_range -- time axis label, nt samples
# f_range -- frequency axis label, nf samples
# y -- spectrogram, nf by nt array
# dbf -- Dynamic range of the spect

def sg_plot( t_range, f_range, y, dbf = 60) :
    eps = 1e-3

    # find maximum
    y_max = abs(y).max()

    # compute 20*log magnitude, scaled to the max
    y_log = 20.0 * np.log10( abs( y ) / y_max + eps )

    fig=plt.figure(figsize=(15,6))

    plt.imshow( np.flipud( 64.0*(y_log + dbf)/dbf ), extent= t_range  + f_range ,cmap=plt.cm.gray, aspect='auto')
    plt.xlabel('Time, s')
    plt.ylabel('Frequency, Hz')
    plt.tight_layout()

def myspectrogram_hann_ovlp(x, m, fs, fc,dbf = 60):
    # Plot the spectrogram of x.
    # First take the original signal x and split it into blocks of length m
    # This corresponds to using a rectangular window %

    isreal_bool = np.isreal(x).all()

    # pad x up to a multiple of m
    lx = len(x);
    nt = (lx + m - 1) // m
    x = np.append(x,zeros(-lx+nt*m))
    x = x.reshape((m/2,nt*2), order='F')
    x = np.concatenate((x,x),axis=0)
    x = x.reshape((m*nt*2,1),order='F')
    x = x[r_[m//2:len(x),np.ones(m//2)*(len(x)-1)].astype(int)].reshape((m,nt*2),order='F')


    xmw = x * np.hanning(m)[:,None];


    # frequency index
    t_range = [0.0, lx / fs]

    if isreal_bool:
        f_range = [ fc, fs / 2.0 + fc]
        xmf = np.fft.fft(xmw,len(xmw),axis=0)
        sg_plot(t_range, f_range, xmf[0:m/2,:],dbf=dbf)
        print 1
    else:
        f_range = [-fs / 2.0 + fc, fs / 2.0 + fc]
        xmf = np.fft.fftshift( np.fft.fft( xmw ,len(xmw),axis=0), axes=0 )
        sg_plot(t_range, f_range, xmf,dbf = dbf)

    return t_range, f_range, xmf

def audio_dev_numbers(p, in_name=u'default', out_name=u'default', debug=False):
    """ din, dout, dusb = audioDevNumbers(p)
    The function takes a pyaudio object
    The function searches for the device numbers for built-in mic and
    speaker and the USB audio interface
    """

    dusb = 'None'
    din = 'None'
    dout = 'None'

    # Linux
    if sys.platform == 'linux2':
        N = p.get_device_count()

        if debug:
            print "Platform: %s"%(sys.platform)
            print "%d devices detected"%(N)

        # Iterate through the devices to find sound card, mic, and speakers
        for n in range(0,N):
           name = p.get_device_info_by_index(n).get('name')
           if debug:
               print "%d: %s"%(n, str(name))
           if 'USB' in name:
               dusb = n
           if in_name in name:
               din = n
           if out_name in name:
               dout = n

    # Mac
    elif sys.platform == 'darwin':
        N = p.get_device_count()
        for n in range(0,N):
            name = p.get_device_info_by_index(n).get('name')
            if name == u'USB PnP Sound Device':
                dusb = n
            if name == u'Built-in Microph':
                din = n
            if name == u'Built-in Output':
                dout = n
    # Windows
    else:
        N = p.get_device_count()
        for n in range(0,N):
            name = p.get_device_info_by_index(n).get('name')
            if name == u'USB PnP Sound Device':
                dusb = n
            if name == u'Microsoft Sound Mapper - Input':
                din = n
            if name == u'Microsoft Sound Mapper - Output':
                dout = n

    if dusb == 'None':
        print('Could not find a usb audio device')
    elif debug:
        print "\nSelected devices:"
        if dusb != 'None':
            dev = p.get_device_info_by_index(dusb)
            print "dusb=%s, input channels=%s, output channels=%s"%(dev['name'],
                    dev['maxInputChannels'], dev['maxOutputChannels'])
        if dout != 'None':
            dev = p.get_device_info_by_index(dout)
            print "dout=%s, input channels=%s, output channels=%s"%(dev['name'],
                    dev['maxInputChannels'], dev['maxOutputChannels'])
        if din != 'None':
            dev = p.get_device_info_by_index(din)
            print "din=%s, input channels=%s, output channels=%s"%(dev['name'],
                    dev['maxInputChannels'], dev['maxOutputChannels'])
        print '\n'

    return din, dout, dusb

def testAudio(play=True, debug=False):
    # create an input output FIFO queues
    Qin = Queue.Queue()
    Qout = Queue.Queue()

    # create a pyaudio object
    p = pyaudio.PyAudio()

    # find the device numbers for builtin I/O and the USB
    din, dout, dusb = audio_dev_numbers(p, in_name='USB', out_name='default', debug=True)
   #p.terminate()
   #return

    # initialize a recording thread. The USB device only supports 44.1KHz sampling rate
    t_rec = threading.Thread(target = record_audio,   args = (Qin,   p, 44100, dusb ))

    # initialize a playing thread.
    #t_play = threading.Thread(target = play_audio,   args = (Qout,   p, 44100, dout ))

    # start the recording and playing threads
    t_rec.daemon = True
    t_rec.start()
    #t_play.start()

    try:
        # record and play about 10 seconds of audio 430*1024/44100 = 9.98 s
        test = np.array([])
        for n in range(0,5):

            samples = Qin.get()

            # You can add code here to do processing on samples in chunks of 1024
            # you will have to implement an overlap an add, or overlap an save to get
            # continuity between chunks
            np.append(test,samples)
            if debug:
                print "n=%d, playing %d samples"%(n, len(samples))

            #Qout.put(samples)
        np.save('test.npy', test)
    except:
        print 'got exception in main thread'
    finally:
        #t_rec.exit()
        p.terminate()
        sys.exit()

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

def genChirpPulse(Npulse, f0, f1, fs, plot=False):
    """Function generates an analytic function of a chirp pulse
    Args:
        Npulse (int): pulse length in samples
        f0 (float): starting frequency of chirp
        f1 (float): end frequency of chirp
        fs (float): sampling frequency
        plot (boolean, optional): plot the time domain output and its spectrogram (default=false)

    Returns:
        numpy.array: analytic function of a chirp pulse
    """
    t = r_[0.0:Npulse]/fs
    f_of_t = np.linspace(f0, f1, len(t))
    phi_of_t = 2*np.pi*np.cumsum(f_of_t)/fs
    chirp = 2*np.sin(phi_of_t)
    #chirp = np.exp(1j* phi_of_t )

    if plot:
        plt.figure(figsize=(15,6))
        plt.plot(np.real(chirp))
        plt.title('Chirp Signal')

        myspectrogram_hann_ovlp(chirp, 128.0, fs, 0.0)
        plt.title('Chirp Spectrogram')
        plt.show()

    return chirp

def fmDemodulate(y, fc, fs, plot=False):
    """FM demodulate the input signal

    Args:
        y (numpy.array): input signal
        fc (float): center frequency (for plotting)
        fs (float): sampling frequency (for plotting)
        plot (boolean, optional): plot the time domain output and its spectrogram (default=false)

    Returns:
        numpy.array: FM demodulated signal
    """
    demod = np.array([ cmath.phase(yn * np.conj( y[n-1]) ) for n, yn in enumerate(y)])
    demod /= (2*np.pi)
    if plot:
        myspectrogram_hann_ovlp(demod, 128.0, fs, fc)

    return demod

def lowpass(x, cutoff, fc, fs=44100.0, plot=False):
    """ Applies an 128 coefficient FIR low pass filter.

    Args:
        x (numpy.array): input signal
        cutoff (float): cutoff frequency of the filter
        fc (float): center frequency (for plotting)
        fs (float, optional): sampling frequency (for plotting, default=44100)
        plot (boolean, optional): plot the time domain output and its spectrogram (default=false)

    Returns:
        numpy.array: low pass filtered signal
    """
    h = signal.firwin(128, cutoff, nyq=fs/2)
    filtered = signal.fftconvolve(h, x)

    if plot:
        myspectrogram_hann_ovlp(filtered, 128.0, fs, fc)
        plt.title('Filtered Spectrogram')
        plt.show()

    return filtered

def text_to_morse(text,fc,fs,dt=60,plot=False):
    """
    Implement a function sig = text2Morse(text, fc, fs,dt). The function will
    take a string and convert it to a tone signal that plays the morse code of
    the text. The function will also take 'fc' the frequency of the tones
    (800-900Hz sounds nice), 'fs' the sampling frequency and 'dt' the morse unit
    time (hence the speed, 50-75ms recommended).
    """
    CODE = {'A': '.-',     'B': '-...',   'C': '-.-.',
        'D': '-..',    'E': '.',      'F': '..-.',
        'G': '--.',    'H': '....',   'I': '..',
        'J': '.---',   'K': '-.-',    'L': '.-..',
        'M': '--',     'N': '-.',     'O': '---',
        'P': '.--.',   'Q': '--.-',   'R': '.-.',
        'S': '...',    'T': '-',      'U': '..-',
        'V': '...-',   'W': '.--',    'X': '-..-',
        'Y': '-.--',   'Z': '--..',

        '0': '-----',  '1': '.----',  '2': '..---',
        '3': '...--',  '4': '....-',  '5': '.....',
        '6': '-....',  '7': '--...',  '8': '---..',
        '9': '----.',

        ' ': ' ', "'": '.----.', '(': '-.--.-',  ')': '-.--.-',
        ',': '--..--', '-': '-....-', '.': '.-.-.-',
        '/': '-..-.',   ':': '---...', ';': '-.-.-.',
        '?': '..--..', '_': '..--.-'
        }

    t = {'.' : r_[0.0:(dt/1000.0)*fs]/fs,
         '-' : r_[0.0:(3*dt/1000.0)*fs]/fs,
         ' ' : r_[0.0:(7*dt/1000.0)*fs]/fs}

    morse = np.array([])
    for char in text:
        sym = CODE[char]
        if sym == ' ':
            morse = np.append(morse, zeros(len(t[' '])))
        else:
            for tone in sym:
                morse = np.append(morse, 0.5*sin(2*pi*t[tone]*fc))
                morse = np.append(morse, np.zeros(len(t['.'])))
            morse = np.append(morse, zeros(len(t['-'])))

    if plot:
        plt.figure(figsize=(15,6))
        plt.plot(np.real(morse))
        plt.title('Morse Signal')

        myspectrogram_hann_ovlp(morse, 128.0, fs*1.0, fc*1.0)
        plt.title('Morse Spectrogram')
        plt.show()

    return morse

def play_audio( data, p, fs, device):
    # data - audio data array
    # p    - pyAudio object
    # fs    - sampling rate
    # device- output device number

    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=device)

    # play audio
    ostream.write( data.astype(np.float32).tostring() )

def sdr_record(sdr, filename, Nsamples=1024000, fc=443.650e6, fs=240000):
    # SDR settings
    sdr.sample_rate = fs    # sampling rate
    sdr.gain = 10           # if the gain is not enough, increase it
    sdr.center_freq = fc

    # Read from SDR and save output
    print 'start'
    data = sdr.read_samples(Nsamples)
    print len(data)
    np.save(filename, data)

def transmit_and_capture(data, outfile, length, title='Captured Data', verbose=False, fs_audio=48000, fs_sdr=240000, fc0=443.650e6,  plot=False):
    """Transmit and receive a signal.
    length seconds
    """
    sdr = RtlSdr()          # Create an RtlSdr object
    p = pyaudio.PyAudio()   # Create a PyAudio object
    Nsamples=256000*length
    fc = fc0*(1.0-85e-6)

    # Get device numbers
    din, dout, dusb = audio_dev_numbers(p, in_name='USB', out_name='default', debug=verbose)

    # Create SDR capture thread as daemon
    capture = threading.Thread(target=sdr_record, args=(sdr, outfile, Nsamples, fc, fs_sdr))
    capture.daemon = True

    # Create play thread as daemon
    play = threading.Thread(target=play_audio, args=(data, p, fs_audio, dusb))
    play.daemon = True

    # Start both threads
    capture.start()
    play.start()

    time.sleep(length+2)

    try:
        if plot:
            print 'Loading data...'
            y = np.load(outfile)
            print 'Generating plot...'
            tt,ff,xmf = myspectrogram_hann_ovlp(y, 256, fs_sdr, fc)
            plt.title(title)
            plt.show()
        else:
            print 'Captured data saved to ' + outfile
    except IOError:
        type, value, traceback = sys.exc_info()
        print('Error loading %s: %s' % (value.filename, value.strerror))
    except Exception as e:
        print 'Error: '+str(e)
    finally:
        print 'Cleaning up...'
        sdr.close()
        p.terminate()
        print 'Closed SDR and PyAudio'

def test_ptt(plen, transmit=False, verbose=False, show_plot=False):
    fs_audio = 48000
    fs_sdr = 240000
    fc0 = 443.650e6
    fc = fc0*(1.0-85e-6)
    filename = 'data/ptt.npy'
    title = 'PTT Signal'

    if transmit:
        ptt =  genPTT(plen, 1000, fs_audio, plot=show_plot)
        print 'Sending PTT...'
        transmit_and_capture(ptt, filename, 4, title, verbose, plot=show_plot)

    try:
        if show_plot:
            print 'Loading PTT data...'
            y = np.load(filename)
            print 'Generating plot...'
            tt,ff,xmf = myspectrogram_hann_ovlp(y, 256, fs_sdr, fc)
            plt.title(title)
            plt.show()

    except IOError:
        type, value, traceback = sys.exc_info()
        print('Error loading %s: %s' % (value.filename, value.strerror))

def test_frequency_response(transmit=False, verbose=False, show_plot=False):
    """
    Demodulate the received FM signal by filtering and taking the derivative of the phase (like in lab 2)
    Low-pass filter the result with a cutoff frequency of 8KHz and decimate by a factor of 16
    Plot the spectrogram of the demodulated signal. Do you see non-linear effects?


    Crop the porsion in which the carrier is active (signal, not noise)
    Compute and plot the frequency response in db. Scale the graph that you can see things well.
    What is the lowest frequency that passes? What is the highest?

    """
    fs_audio = 48000
    fs_sdr = 240000
    fc0 = 443.650e6
    fc = fc0*(1.0-85e-6)
    filename = 'data/received_chirp.npy'
    title = 'Received Chirp'

    if transmit:
        # Send a 5 second chirp pulse
        ptt =  genPTT(150, 400, fs_audio)
        chirp = np.append(ptt, genChirpPulse(5*fs_audio, 0, 8000, fs_audio))
        print 'Sending chrip'
        transmit_and_capture(chirp, filename, 6, title, verbose, plot=show_plot)

    try:
        print 'Loading received chirp data...'
        y = np.load(filename)
        print 'Generating plot...'
        tt,ff,xmf = myspectrogram_hann_ovlp(y, 256, fs_sdr*1.0, fc*1.0)
        plt.title(title)

        # FM Demodulate the raw signal
        demod = np.angle(y*np.conj(np.roll(y,1)))

        # Low pass filter fc = 8 kHz
        h = signal.firwin(256, 8000.0, nyq=fs_sdr/2.0)
        filtered = signal.fftconvolve(h, demod)

        # Downsample by M=16
        dec = filtered[::16]

        myspectrogram_hann_ovlp(dec, 128.0, fs_sdr/16.0, fc*1.0)
        plt.title(title+': Demodulated and Decimated')

        plt.figure(figsize=(15,6))
        plt.semilogy( np.linspace(0, 8000, len(dec)/2), np.abs(  np.fft.fft(dec)[:len(dec)/2] ) )
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('dB')
        plt.title(title+'\nSpectrum')

        if show_plot:
            plt.show()
    except IOError:
        type, value, traceback = sys.exc_info()
        print('Error loading %s: %s' % (value.filename, value.strerror))

def test_morse(text='KK6KKT', transmit=False, verbose=False, show_plot=False):
    fs_audio = 48000
    fs_sdr = 240000
    fc0=443.650e6
    fc = fc0*(1.0-85e-6)
    filename = 'data/morse.npy'
    title = 'Morse Code'

    if transmit:
        ptt =  genPTT(150, 400, fs_audio)
        morse = np.append(ptt, text_to_morse(text, fc, fs_audio, dt=75))
        transmit_and_capture(morse, filename, 6, title, verbose, plot=show_plot)

    try:
        print 'Loading data...'
        y = np.load(filename)
        print 'Generating plot...'
        tt,ff,xmf = myspectrogram_hann_ovlp(y, 256, fs_sdr, fc)
        plt.title(title)

        if show_plot:
            plt.show()
    except IOError:
        type, value, traceback = sys.exc_info()
        print('Error loading %s: %s' % (value.filename, value.strerror))

def main():
    parser = argparse.ArgumentParser(description='Run lab 3 functions.')
    parser.add_argument('function', choices=['ptt', 'response', 'morse', 'all'], help='pick a function to run')
    parser.add_argument('plen', type=float, default=150.0, help='ptt length')
    parser.add_argument('text', type=str, default='KK6KKT', help='text to send in morse')
    parser.add_argument('-t', '--transmit', action='store_true', default=False, help='print additional output')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print additional output')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='show plots')
    args = parser.parse_args()

    if args.function == 'ptt':
        test_ptt(args.plen, transmit=args.transmit, verbose=args.verbose,
                show_plot=args.plot)
    elif args.function == 'response':
        test_frequency_response(transmit=args.transmit, show_plot=args.plot)
    elif args.function == 'morse':
        test_morse(transmit=args.transmit, show_plot=args.plot)
    else:
        test_ptt(args.plen, transmit=args.transmit, verbose=args.verbose,
                show_plot=args.plot)
        test_frequency_response(transmit=args.transmit, show_plot=args.plot)
        test_morse(transmit=args.transmit, show_plot=args.plot)

    print 'Exiting'

if __name__ == "__main__":
    main()
