from nader_func import *
from qam import *



prefix = np.array([[0],[2],[10],[8],[6],[11] , [0]])
bits= np.array([[0],[2],[10],[8],[6],[11] , [0], [4], [2],[1], [2], [5], [6], [8], [6], [10], [3], [14],[0],[10]])
randombits = randomBits(16)
#bits = prefix
#bits = np.array([0,2,10,8,5,11,0,5,7,13,9])
#bits = np.concatenate(bits, prefix)
fs = 44100
baud = 300
Nbits = len(bits)
Ns = fs/baud
f0=1800
N = Nbits * Ns
code = np.array((-2-2j, -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j,+2+1j,1-2j,+1-1j,1+2j,1+1j))/2


M, t = makeSymbols(bits, fs, code, Ns)


#This is the fully modulated bit string to be sent:
QAM = (M.real.ravel()*cos(2*pi*f0*t) - M.imag.ravel()*sin(2*pi*f0*t))/2/sqrt(2)


# fig = figure(figsize = (16,4))
# plot(t,QAM.real)
# plot(t,QAM.imag)
# xlabel('time [s]')
# title("QAM=16 of the sequence:"+ np.array2string(transpose(bits)))

# fig = figure()
# plot(M.real.ravel()*cos(2*pi*f0*t))
#fig = figure()
#scatter(np.real(M), np.imag(M))
#show()
#print code[bits]

#begin demodulation:
r,i = demod(QAM, f0, t)

lp = signal.firwin(100,pi*f0/4,nyq=44100.0/2.0)
r_lp = signal.fftconvolve(lp,r)[50:]
i_lp = signal.fftconvolve(lp, i)[50:]


frange = np.linspace(-fs/2,fs/2,len(r))
frange_filt = np.linspace(-fs/2,fs/2,len(r_lp))

# fig = figure()
# plot(M.real.ravel())
# plot(r_lp)
# show()


M_prefix, t_prefix = makeSymbols(prefix, fs, code, Ns)
fig = figure()
plot(M_prefix.real.ravel())
plot(r_lp.ravel())
title("real")
fig = figure()

plot(M_prefix.imag.ravel())
plot(i_lp.ravel())
title("imaginary")
show()
corr_r = signal.correlate(M_prefix.real.ravel(), r_lp.ravel())
corr_i = signal.correlate(M_prefix.imag.ravel(), i_lp.ravel())

fig = figure()
plot(corr_r)
plot(corr_i)
title("corr")
show()
