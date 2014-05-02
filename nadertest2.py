from nader_func import *
from qam import *



prefix = np.array([[0],[2],[10],[8]])
bits= np.array([[1],[1],[1],[1],[1],[0],[2],[10],[8],[6],[11] , [0], [4], [2],[1], [2], [5], [6], [8], [6], [10], [3], [14],[0],[10]])
randombits = randomBits(16)

fs = 44100
baud = 300
Nbits = len(bits)
Ns = fs/baud
f0=1800
N = Nbits * Ns
code = np.array((-2-2j, -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j,+2+1j,1-2j,+1-1j,1+2j,1+1j))/2

# modulate both the bits and the prefix, so you can maybe correlate
# with modulated prefix:
M, t = makeSymbols(bits, fs, code, Ns)
M_prefix, t_prefix = makeSymbols(prefix, fs, code, Ns)

#This is the fully modulated bit string to be sent:
QAM = (M.real.ravel()*cos(2*pi*f0*t) - M.imag.ravel()*sin(2*pi*f0*t))/2/sqrt(2)

#Do it to the prefix also:
QAM_prefix = (M_prefix.real.ravel()*cos(2*pi*f0*t_prefix) - M_prefix.imag.ravel()*sin(2*pi*f0*t_prefix))/2/sqrt(2) 

#modulated correlation (of sunusoids):
corr = np.correlate(QAM,QAM_prefix,"same")
corr_index = argmax(abs(corr))
print "index from corr: ", corr_index

# fig = figure()
# plot(abs(corr))
# title("modulated sinusoid corr")

# -----------------
#begin demodulation:
r,i = demod(QAM, f0, t)


#lowpass the signals to remove images
lp = signal.firwin(100,pi*f0/4,nyq=44100.0/2.0)
r_lp = signal.fftconvolve(lp,r)[50:]
i_lp = signal.fftconvolve(lp, i)[50:]


frange = np.linspace(-fs/2,fs/2,len(r))
frange_filt = np.linspace(-fs/2,fs/2,len(r_lp))


corr_r = abs(signal.correlate(r_lp.ravel(),M_prefix.real.ravel(),"same"))
corr_i = abs(signal.correlate(i_lp.ravel(), M_prefix.imag.ravel(),"same"))

corr_index = argmax(corr_r)
print "index from corr: ", corr_index 

# fig = figure()
# plot(corr_r)
# plot(corr_i)
# title("demodulated bits corr")
# show()


#This next block of code is supposed to trim everything that isn't
#what we want from the front (afte corr), but I don't think it 
#is 100% working. Close, though, off by a bit or two?
r = r_lp[corr_index:]

i = i_lp[corr_index:]

fig = figure()
plot(r)
#5 is the number of garbage symbols i put in before the prefix in the signal
plot(M.real.ravel()[Ns*5:])
show()

#need to know number of bits for this to work (packet size i guess?)
#Nbits might be wrong here, since it really needs the nubmer of bits
#minus the amount you chopped off after correlating
idx = r_[0:len(r):len(r)/Nbits]
r = r/max(r)*max(M.real.ravel())*2 #change this 2 depending on the input amplitude
i = i/max(i)*max(M.imag.ravel())*2 #change this 2 depending on the input amplitude
r = np.around(r)
i = np.around(i)
r_dec = np.around(r[idx])
i_dec = np.around(i[idx])
offset = len(M.real.ravel())/Nbits

print "len(r)", len(r)
print "len(r_dec)", len(r_dec)

r_avg = []
i_avg = []
periods = r_[0:len(r):offset]


for prd in periods[0:-1]:
    current_avg_r = np.average(r[prd:prd+offset])
    r_avg = np.append(r_avg, current_avg_r)
    
    current_avg_i = np.average(i[prd:prd+offset])
    i_avg = np.append(i_avg, current_avg_i)
    

decoded = np.around(r_avg) + 1j*np.around(i_avg) 

finalbits = np.array([])
decoded = decoded/2

