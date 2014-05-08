#!/usr/bin/env python
import numpy as np
import pyaudio
import sys
import argparse
import Queue
import threading
from scipy import signal

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
            print name
            if 'USB' in name:
                dusb = n
            if in_name in name:
                din = n
            if out_name in name:
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

def play_audio(Q, p, fs , dev):
    '''
    play_audio plays audio with sampling rate = fs
    Q - A queue object from which to play
    p   - pyAudio object
    fs  - sampling rate
    dev - device number
    '''

    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    count = 0
    while (1):
        data = Q.get()
        count += len(data)
        if data=="EOT":
            print "Finished playing"
            print "Total %d samples"%(count)
            ostream.close()
            Q.task_done()
            break
        try:
            ostream.write( data.astype(np.float32).tostring() )
            Q.task_done()
        except Exception as e:
            print e
            ostream.close()
            break

def gen_ptt(plen=150, zlen=400, fs=44100.0, plot=False):
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

    t = np.r_[0.0:(plen/1000.0)*fs]/fs
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

def transmit(data, fs=44100.0, verbose=False):
    '''
    Args:
        data (numpy.array): data to send
    '''
    p = pyaudio.PyAudio()
    din, dout, dusb = audio_dev_numbers(p, in_name=u'default',
            out_name=u'default', debug=verbose)

    # Array to queue
    Qout = Queue.Queue()
    ptt = gen_ptt(plen=100, zlen=2200, fs=fs)
    data = np.append(ptt, data)
    print "sending %d samples"%(len(data))
   #for n in np.r_[0:len(data):512]:
   #    Qout.put(data[n:n+512])
    Qout.put(data)
    Qout.put("EOT")

    send = threading.Thread(target=play_audio, args=(Qout, p, fs, dusb))
    send.daemon = True
    send.start()

    # Block until all received data finishes playing
    Qout.join()
    print 'Terminating PyAudio...'
    p.terminate()

def rand_bits(fs):
    baud = 300  # symbol rate
    Ns = fs/baud
    f0 = 1800

    code = np.array((-2-2j,
        -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j+2+1j,1-2j,+1-1j,1+2j,1+1j))/2

    np.random.seed(seed=1)
    rbits = np.int16(np.random.rand(6,1)*15)
    prefix = np.array([[0],[2],[10],[8]])
    bits = np.int16(np.random.rand(26,1)*15)

    Nbits = len(rbits) + len(prefix) + len(bits)  # number of bits
    bits = np.array(rbits.tolist() + prefix.tolist() + bits.tolist())
    N = Nbits * Ns

    M = np.tile(code[bits],(1,Ns))
    t = np.r_[0.0:N]/fs

    np.save('data/real.npy', M.real.ravel())
    np.save('data/imag.npy', M.imag.ravel())

    QAM = (M.real.ravel()*np.cos(2*np.pi*f0*t) -
            M.imag.ravel()*np.sin(2*np.pi*f0*t))/2/np.sqrt(2)

    return QAM

def prefix(fs, baud=300, shape=False):
    Ns = fs/baud
    f0 = 1800

    code = np.array((-2-2j,
        -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j+2+1j,1-2j,+1-1j,1+2j,1+1j))/2

    prefix = np.array([[0],[2],[10],[8]])

    Nbits = len(prefix)# number of bits
    bits = np.array(prefix.tolist())
    N = Nbits * Ns
    t = np.r_[0.0:N]/fs

    if shape:
       #x = np.r_[-2,2:(1.0/147)]
       #h = np.sinc(x)*signal.hann(147*4)
       #impulses = np.zeros(len(bits)*Ns)
       #for i,b in enumerate(bits):
       #    impulses[i*Ns]=code[int(b)]
       #M = signal.fftconvolve(impulses, h)
       #t = np.r_[0.0:len(M)]/fs

        imp = np.zeros(N,dtype='complex')
        imp[::Ns] = code[bits].ravel()
        h = signal.firwin(Ns*4,1.0/Ns)
        imp_sinc = signal.fftconvolve(imp,h,mode='full')
        t = np.r_[0.0:len(imp_sinc)]/fs
        #QAM_s = #imp_sinc*np.exp(1j*2*np.pi*f0*t))
        #QAM = M*np.exp(1j*2*np.pi*f0*t)/2/np.sqrt(2)
        QAM = (imp_sinc.real*np.cos(2*np.pi*f0*t) -
                imp_sinc.imag*np.sin(2*np.pi*f0*t))/2/np.sqrt(2)
        print sum(QAM.imag)
        QAM = QAM.real
        np.save('data/prefix_real_%d_.npy'%(baud), imp_sinc.real)
        np.save('data/prefix_imag_%d_.npy'%(baud), imp_sinc.imag)
        return QAM
    else:
        M = np.tile(code[bits],(1,Ns))
        QAM = (M.real.ravel()*np.cos(2*np.pi*f0*t) -
                M.imag.ravel()*np.sin(2*np.pi*f0*t))/2/np.sqrt(2)
        np.save('data/prefix_real_%d_.npy'%(baud), M.real.ravel())
        np.save('data/prefix_imag_%d_.npy'%(baud), M.imag.ravel())
        return QAM

def main():
    parser = argparse.ArgumentParser(description='Record data from the radio.')
    parser.add_argument('--filename', default=None, help='.npy file with the data to send')
    parser.add_argument('--fs', type=float, default=48000.0, help='Sampling frequency to send data')
    parser.add_argument('-b', '--baud', type=int, default=300, help='Symbol rate')
    parser.add_argument('--prefix', action='store_true', default=False, help='Send prefix only')
    parser.add_argument('--shape', action='store_true', default=False,
            help='Shape pulses')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print additional output')
    args = parser.parse_args()

    if args.prefix:
        test = np.append(prefix(args.fs, args.baud), np.zeros(args.fs))
        #test = np.append(test, prefix(args.fs, args.baud, args.shape))
        print 'signal length: %d'%(len(test))
        print 'Symbol rate: %d'%(args.baud)
        print 'Sample rate: %d'%(args.fs)

        transmit(test, fs=args.fs, verbose=args.verbose)

    elif args.filename:
        transmit(np.load(args.filename), fs=args.fs, verbose=args.verbose)

    else:
        transmit(rand_bits(args.fs), fs=args.fs, verbose=args.verbose)

if __name__ == "__main__":
    main()
