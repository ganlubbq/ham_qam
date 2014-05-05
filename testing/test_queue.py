import numpy as np
import time
import pyaudio
import sys

from Queue import Queue
from threading import Thread
from scipy.io.wavfile import write as wav_write
from scipy.io.wavfile import read as wav_read

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
#TODO add silence to prevent buffer underrun
# http://stackoverflow.com/questions/19230983/prevent-alsa-underruns-with-pyaudio
def play_audio(Q, p, fs , dev, Qsav):
    # play_audio plays audio with sampling rate = fs
    # Q - A queue object from which to play
    # p   - pyAudio object
    # fs  - sampling rate
    # dev - device number

    # Example:
    # fs = 44100
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # Q = Queue.queue()
    # Q.put(data)
    # Q.put("EOT") # when function gets EOT it will quit
    # play_audio( Q, p, fs,1 ) # play audio
    # p.terminate() # terminate pyAudio

    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    while (1):
        data = Q.get()
        if data=="EOT" :
            print data
            #Qsav.put(data)
            ostream.close()
            Q.task_done()
            break
        try:
            ostream.write( data.astype(np.float32).tostring() )
            Qsav.put(data.astype(np.float32))
            Q.task_done()
        except Exception as e:
            print e
            ostream.close()
            break

def record_audio(Qin, Qout, p, fs ,dev,chunk=512):
    # record_audio records audio with sampling rate = fs
    # queue - output data queue
    # p     - pyAudio object
    # fs    - sampling rate
    # dev   - device number
    # chunk - chunks of samples at a time default 1024
    #
    # Example:
    # fs = 44100
    # Q = Queue.queue()
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # record_audio( Q, p, fs, 1) #
    # p.terminate() # terminate pyAudio


    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    frames = [];

    while (1):
        try:  # when the pyaudio object is distroyed stops
            data_str = istream.read(chunk) # read a chunk of data
        except Exception as e:
            print "error reading input"
            print e
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        try:
            Qin.get_nowait() # append to list
            print "Stop recording"
            istream.close()
            Qout.put("EOT")
            return
        except:
            print "put %d samples"%(len(data_flt))
            Qout.put(data_flt) # append to list

def save_wav(Qsav, filename, fs):
    data = np.array([]);
    print '\ngetting data\n'
    while(1):
        chunk = Qin.get() # append to list
        if chunk=="EOT":
            print 'Saving output...'
            wav_write(filename, fs, data)
            Q.task_done()
            return
        else:
            data = np.append(data, chunk) # append to list
            print len(data)
            Q.task_done()

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

    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)

fs = 44100.0
p = pyaudio.PyAudio() #instantiate PyAudio
ptt = gen_ptt(plen=400, zlen=450, fs=fs)
din, dout, dusb =  audio_dev_numbers(p, in_name=u'default', out_name=u'default',
        debug=False)
print "din sample rate: %f"%(p.get_device_info_by_index(din)['defaultSampleRate'])
print "dout sample rate: %f"%(p.get_device_info_by_index(dout)['defaultSampleRate'])
print "dusb sample rate: %f"%(p.get_device_info_by_index(dusb)['defaultSampleRate'])

Qin = Queue()
Qout = Queue()
Qsav = Queue()

play = Thread(target=play_audio, args=(Qout, p, fs, dout, Qsav))
play.daemon = True
play.start()

record = Thread(target=record_audio, args=(Qin, Qout, p, fs, dusb))
record.daemon = True
record.start()

time.sleep(5)
Qin.put("EOT")
Qout.join()       # block until all tasks are done
print '\nterminating\n'
p.terminate()

print '\ngetting data\n'
#output = np.matrix([item for item in Qsav.queue])
#output = np.matrix([item for item in Qsav.queue])

# Qsav to array
output = []
for chunk in Qsav.queue:
    print np.shape(chunk)
    output+=chunk.tolist()
output = np.array(output)
print len(output)

# Array to queue
Qtest = Queue()
Qout = Queue()
print '\nloading\n'
for n in np.r_[0:len(output):512]:
    Qout.put(output[n:n+512])
Qout.put("EOT")

time.sleep(2)
p = pyaudio.PyAudio() #instantiate PyAudio
print '\nplaying\n'
play = Thread(target=play_audio, args=(Qout, p, fs, dout, Qtest))
play.daemon = True
play.start()
Qout.join()       # block until all tasks are done
p.terminate()
print '\nexit\n'

print '\nwriting output file\n'
#wav_write('test.wav', 44100, output)
np.save('test.npy',output)

p = pyaudio.PyAudio() #instantiate PyAudio
print '\nreading input file\n'
#fs, test = wav_read('test.wav')
test = np.load('test.npy')

Qout = Queue()
Qsav = Queue()

print '\nloading\n'
for n in np.r_[0:len(test):512]:
    Qout.put(test[n:n+512])
Qout.put("EOT")

print '\nplaying\n'
play = Thread(target=play_audio, args=(Qout, p, fs, dout, Qsav))
play.daemon = True
play.start()
Qout.join()       # block until all tasks are done
p.terminate()
print '\nexit\n'
quit()

save = Thread(target=save_wav, args=(Qsav, 'test.wav', fs))
save.daemon = True
save.start()
Qsav.join()       # block until all tasks are done



