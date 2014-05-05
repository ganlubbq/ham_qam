#!/usr/bin/env python
import numpy as np
import time
import pyaudio
import sys
import argparse
import Queue
import threading

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

def play_audio(Q, p, fs , dev, Qsav):
    '''
    play_audio plays audio with sampling rate = fs
    Q - A queue object from which to play
    p   - pyAudio object
    fs  - sampling rate
    dev - device number

    Example:
    fs = 44100
    p = pyaudio.PyAudio() #instantiate PyAudio
    Q = Queue.queue()
    Q.put(data)
    Q.put("EOT") # when function gets EOT it will quit
    play_audio( Q, p, fs,1 ) # play audio
    p.terminate() # terminate pyAudio
    '''

    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    count = 0
    while (1):
        data = Q.get()
        if data=="EOT" :
            print "Finished playing"
            print "received %d samples"%(count)
            #Qsav.put(data)
            ostream.close()
            Q.task_done()
            break
        try:
            ostream.write( data.astype(np.float32).tostring() )
            count += len(data)
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
    count = 0

    while (1):
        try:  # when the pyaudio object is distroyed stops
            data_str = istream.read(chunk) # read a chunk of data
        except Exception as e:
            print "error reading input"
            print e
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        count += len(data_flt)
        try:
            Qin.get_nowait() # append to list
            Qout.put(data_flt) # append to list
            print "Stop recording"
            istream.close()
            Qout.put("EOT")
            print "got %d"%(count)
            return
        except:
            Qout.put(data_flt) # append to list

def timed_record(filename, length, fs):
    '''
    length (float) time to record in seconds
    '''
    p = pyaudio.PyAudio()
    din, dout, dusb = audio_dev_numbers(p, in_name=u'default',
            out_name=u'default', debug=False)

    Qin = Queue.Queue()
    Qout = Queue.Queue()
    Qsav = Queue.Queue()

    # Create and start a thread to play everything received from the radio
    play = threading.Thread(target=play_audio, args=(Qout, p, fs, dout, Qsav))
    play.daemon = True
    play.start()

    # Create and start a thread to receive data from the radio for a specified
    # amount of time
    record = threading.Thread(target=record_audio, args=(Qin, Qout, p, fs, dusb))
    record.daemon = True
    record.start()

    time.sleep(length)
    Qin.put("EOT")

    # Block until all received data finishes playing
    Qout.join()
    print 'Terminating PyAudio...'
    p.terminate()

    if filename:
        # Save received data to an npy file
        print 'Saving to data/%s.npy...'%(filename)
        output = []
        for chunk in Qsav.queue:
            output+=chunk.tolist()
        print len(output)
        output = np.array(output)
        np.save('data/'+filename, output)

def manual_record(filename, fs):
    '''
    length (float) time to record in seconds
    '''
    p = pyaudio.PyAudio()
    din, dout, dusb = audio_dev_numbers(p, in_name=u'default',
            out_name=u'default', debug=False)

    Qin = Queue.Queue()
    Qout = Queue.Queue()
    Qsav = Queue.Queue()

    # Create and start a thread to play everything received from the radio
    play = threading.Thread(target=play_audio, args=(Qout, p, fs, dout, Qsav))
    play.daemon = True
    play.start()

    # Create and start a thread to receive data from the radio for a specified
    # amount of time
    record = threading.Thread(target=record_audio, args=(Qin, Qout, p, fs, dusb))
    record.daemon = True
    record.start()

    raw_input('\nPress ENTER to stop recording')
    Qin.put("EOT")

    # Block until all received data finishes playing
    Qout.join()
    print 'Terminating PyAudio...'
    p.terminate()

    if filename:
        # Save received data to an npy file
        print 'Saving to data/%s.npy...'%(filename)
        output = []
        for chunk in Qsav.queue:
            output+=chunk.tolist()
        print len(output)
        output = np.array(output)
        np.save('data/'+filename, output)

def main():
    parser = argparse.ArgumentParser(description='Record data from the radio.')
    #parser.add_argument('function', choices=['ptt', 'response', 'morse', 'all'], help='pick a function to run')
    parser.add_argument('--filename', default=None, help='output to data/FILENAME.npy')
    parser.add_argument('-t', '--time', type=float, default=0,
            help='Perform a timed recording for TIME seconds')
    parser.add_argument('--fs', type=float, default=48000.0, help='print additional output')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print additional output')
    args = parser.parse_args()

    if args.time > 0:
        timed_record(args.filename, args.time, args.fs)
    else:
        manual_record(args.filename, args.fs)

if __name__ == "__main__":
    main()
