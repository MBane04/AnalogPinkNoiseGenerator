"""
Eric J. Wyers
wyers@tarleton.edu
ELEN 3445
Analog Pink Noise Generator Project

pinknoise_transient_testbench.py
"""

# this code comes with no warranty or guarantee of any kind

# this script does least-squares fitting on the fft-based psd of the pink 
# noise generator transient response and produces several error results;
# see pinknoise_transient_testbench.asc

# this script computes the transient performance score by comparing simulated 
# transient psd to the ideal pink noise psd; the lower the score, the better

# ltspice testbench details; the following spice directives were used to 
# generate the pn_5sec_data.wav file:
# .param tstart=1 tstop=6 fs_op=50k
# .options plotwinsize=0 numdgt=15 measdgt=15
# .tran 0 {tstop} {tstart} {1/fs_op}
# .wave pn_5sec_data.wav 16 50k V(vout)
# the white noise input to the gain stage is a voltage source defined as:
# wavefile="white_python.wav" chan=0

import numpy as np
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def eng_str(x):
    from math import log10
    y = abs(x)
    if (y==0):
        z = 0
        engr_exponent = 0
        sign = ''
    else:
        exponent = int(log10(y)) 
        engr_exponent = exponent - exponent%3
        z = y/10**engr_exponent
        if (z>=0.1) and (z<1):
            z=z*1e3
            engr_exponent=engr_exponent-3
        sign = '-' if x < 0 else ''
    if (engr_exponent == 0):
        return sign+str(z)
    else:
        return sign+str(z)+'e'+str(engr_exponent)


# convenience parameters for saving plots to pdf, selecting .txt file, model
# selection, and debug results reporting and plotting
figsave = 0 #set to 1 to save figures; else 0
useltspice = 0 #set to 1 to use ltspice data; else 0 for audiocheck.net data
model_select = 4 #set to 4 for 4th-order model; else 3 for 3rd-order model
debug = 0 #set to 1 to dump out various intermediate results; else 0

# additional parameters for psd processing and modeling
fmin = 20.0 #audio lower-band edge 
fmax = 20e3 #audio upper-band edge
deltaf = 1e-3 #minimum frequency adjustment parameter for log sampling
bins_per_decade_meas = 6 #set bins per decade for measured psd

if (debug == 1):
    print('figsave:')
    print(figsave)
    print('useltspice:')
    print(useltspice)
    print('model_select:')
    print(model_select)

# set the value of p for the lp-norm
p = 2 #2-norm error

# set the transient performance score number of decimal places
dec_place = 2 #performance score represented to this many decimal places

# import the pink noise .wav file
if (useltspice == 0):
    # 10-second duration, high-quality pink noise sample from audiocheck.net
    sampling_rate, audio_data_int = wavfile.read('audiocheck.net_pinknoise.wav')
elif (useltspice == 1):
    # 5-second duration pink noise sample from LTSpice
    sampling_rate, audio_data_int = wavfile.read('pn_5sec_data.wav') 

# if the audio is stereo, convert it to mono by taking the average of channels
if len(audio_data_int.shape) > 1:
    audio_data_int = np.mean(audio_data_int, axis=1)

# determine the maximum possible amplitude based on the data type
# for example, int16 has a max value of 32767
if audio_data_int.dtype == np.int16:
    max_amplitude = 2**15 - 1  #for 16-bit signed integers
elif audio_data_int.dtype == np.int32:
    max_amplitude = 2**31 - 1  #for 32-bit signed integers
elif audio_data_int.dtype == np.uint8:
    # 8-bit unsigned data is typically centered around 128
    audio_data_int = audio_data_int.astype(np.int16) - 128
    max_amplitude = 2**7 - 1 #for 8-bit unsigned, after centering
else:
    # handle other integer types or raise an error
    raise TypeError(f"unsupported .wav data type: {audio_data_int.dtype}")

audio_data = audio_data_int.astype(np.float64) / max_amplitude

# plot the time-domain data
T = 1/sampling_rate
tmin = 0.
tmax = (len(audio_data)-1)*T
if (useltspice == 0):
    tvec = np.arange(tmin, tmax+T, T)
else:
    tvec = np.arange(tmin, tmax, T)

plt.figure(figsize=(10, 6))
plt.plot(tvec, audio_data)
plt.title('pink noise voltage in the time domain')
plt.xlabel('time [s]')
plt.ylabel('voltage [V]')
plt.grid(True)
if (figsave == 1):
    if (useltspice == 0):
        plt.savefig('pinknoise_transient_testbench_audiocheck_plot1.pdf')
    else:
        plt.savefig('pinknoise_transient_testbench_ltspice_plot1.pdf')
plt.show()

# plot the histogram of the noise signal
plt.hist(audio_data, bins=30, edgecolor='black')
plt.xlabel("value")
plt.ylabel("frequency")
plt.title("histogram of pink noise signal")
if (figsave == 1):
    if (useltspice == 0):
        plt.savefig('pinknoise_transient_testbench_audiocheck_plot2.pdf')
    else:
        plt.savefig('pinknoise_transient_testbench_ltspice_plot2.pdf')
plt.show()

maxmeasdata = np.max(audio_data)
minmeasdata = np.min(audio_data)

# check if design is valid or not based on the signs of the voltage peaks
if (maxmeasdata/minmeasdata > 0):
    invalid_design = 1 #this means that max output and min output have the same signs
else:
    invalid_design = 0 #this means that max output is positive and min output is negative

# now conpute the psd of data via the fft
N = len(audio_data)
Dfft = np.fft.rfft(audio_data,N) #for N even, Dfft should be N/2 + 1 in length
NDfft = len(Dfft)
Ffftmin = 0.
Ffftmax = 1.
Ffftstep = (Ffftmax - Ffftmin)/(NDfft - 1)
frequencies = (sampling_rate/2)*np.arange(Ffftmin, Ffftmax+Ffftstep, Ffftstep)

psd = (1/(sampling_rate*N))*np.abs(Dfft)**2
psd[1:-2] = 2*psd[1:-2]

# save a copy of the psd outputs above for plot3 (plots are indexed to 1)
freqmeasplot2 = frequencies.copy()
psdmeasplot2 = psd.copy()

# define logarithmic bin edges
min_freq = frequencies[1] - deltaf #start from a small positive frequency to avoid log(0)
max_freq = frequencies[-1]
num_dec = np.log10(max_freq/min_freq)
log_bins = np.logspace(np.log10(min_freq), np.log10(max_freq), num=int(num_dec * bins_per_decade_meas))

# create a DataFrame for easier binning
df = pd.DataFrame({'frequency': frequencies, 'power': psd})
# use pandas.cut to assign each linear frequency to a log bin
df['log_bin'] = pd.cut(df['frequency'], bins=log_bins, labels=False, include_lowest=True)
# group by log_bin and calculate the mean power
psd = np.array(df.groupby('log_bin')['power'].mean())
frequencies = (log_bins[1:] + log_bins[:-1]) / 2 #use the center of each bin for plotting

freqSize = len(frequencies)
kflaglow = 0
kflaghigh = 0
klow = 0
khigh = freqSize - 1
fminpsd = frequencies[klow]
fmaxpsd = frequencies[khigh]

for k in range(freqSize):
    if (frequencies[k] >= fmin)&(kflaglow == 0):
        klow = k
        fminpsd = frequencies[klow]
        kflaglow = 1
    if (frequencies[k] > fmax)&(kflaghigh == 0):
        khigh = k - 1
        fmaxpsd = frequencies[khigh]
        kflaghigh = 1

if (debug == 1):
    print('freqSize:')
    print(freqSize)
    print('klow:')
    print(klow)
    print('kflaglow:')
    print(kflaglow)
    print('khigh:')
    print(khigh)
    print('kflaghigh:')
    print(kflaghigh)
    print('fminpsd:')
    print(eng_str(fminpsd))
    print('fmaxpsd:')
    print(eng_str(fmaxpsd))

# get frequencies from fmin=20 Hz to fmax=20 kHz
freqvec = frequencies[klow:khigh+1]
psdvec = 10*np.log10(psd[klow:khigh+1])

# save a copy of these arrays for plot3 (plots are indexed to 1)
freqvecplot_meas = freqvec.copy()
psdvecplot_meas = psdvec.copy()

# now set up the 4th- or 3rd-order least-squares (LS) model-fitting problem
Acol3 = np.log10(freqvec).reshape(-1,1)
Acol0 = Acol3**4
Acol1 = Acol3**3
Acol2 = Acol3**2
Acol4 = np.ones((len(Acol1),1))
if (model_select == 4):
    A = np.hstack((Acol0,Acol1,Acol2,Acol3,Acol4))
elif (model_select == 3):
    A = np.hstack((Acol1,Acol2,Acol3,Acol4))
d = psdvec.reshape(-1,1)
# now solve the Ax = d least-squares problem
xls = np.linalg.solve(A.T@A,A.T@d)

if (model_select == 4):
    a4ls = xls[0]
    a3ls = xls[1]
    a2ls = xls[2]
    a1ls = xls[3]
    a0ls = xls[4]
elif (model_select == 3):
    a4ls = xls[0]*0.
    a3ls = xls[0]
    a2ls = xls[1]
    a1ls = xls[2]
    a0ls = xls[3]

if (debug == 1):
    print('a4ls:')
    print(eng_str(a4ls[0]))
    print('a3ls:')
    print(eng_str(a3ls[0]))
    print('a2ls:')
    print(eng_str(a2ls[0]))
    print('a1ls:')
    print(eng_str(a1ls[0]))
    print('a0ls:')
    print(eng_str(a0ls[0]))

psdls4 = A@xls.reshape(-1,1)

# note: I reserve the right to change the model above to higher than 4th 
# order; e.g., I may decide that a 5th-order model is better at capturing
# key performance metrics; and it may be the case that I decide I want to 
# use a lower-order model, too, such as a 3rd- or a 2nd-order model

# note that psd responses which are more linear will/should have relatively
# small 4th-, 3rd-, and 2nd-order model coefficients, a4ls, a3ls, and a2ls, 
# respectively; stated another way, designs which yield relatively large 
# a4ls, a3ls, and/or a2ls values will be penalized due to having relatively 
# large max psd error performance (and yet still may also have suboptimal 
# slope error, too)

# number of samples per decade
num_samp_per_decade = 1000

# calculate the number of decades over the frequency range of interest
num_decades = np.log10(fmax/fmin)

# total number of samples needed
total_samples = int(num_samp_per_decade * num_decades)

# generate logarithmically-spaced samples
freq_list = np.logspace(
    start=np.log10(fmin),
    stop=np.log10(fmax),
    num=total_samples,
    base=10,
    endpoint=True
)

freqlistSize = len(freq_list)

psdls4dense = np.zeros([freqlistSize,1])

for k in range(freqlistSize):
    f = freq_list[k]
    f1 = np.log10(f)
    f2 = f1**2
    f3 = f1**3
    f4 = f1**4
    psdls4dense[k]  = a4ls*f4 + a3ls*f3 + a2ls*f2 + a1ls*f1 + a0ls

# plot the PSDs of the data and the densely-sampled model (f=0 data not shown)
plt.figure(figsize=(10, 6))
plt.semilogx(freqvecplot_meas, psdvecplot_meas, 'b') 
plt.semilogx(freq_list, psdls4dense, 'k--')
plt.semilogx(freqmeasplot2[1:], 10*np.log10(psdmeasplot2[1:]), 'r', linewidth=0.02)
plt.title('power spectral density of data and densely-sampled model')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend(['data psd', 'dense LS fit psd', 'fft-based psd output'])
plt.grid(True)
if (figsave == 1):
    if (useltspice == 0):
        plt.savefig('pinknoise_transient_testbench_audiocheck_plot3.pdf')
    else:
        plt.savefig('pinknoise_transient_testbench_ltspice_plot3.pdf')
plt.show()

# now set up 1st-order least-squares (LS) fitting problem
Acol1 = np.log10(freq_list).reshape(-1,1)
Acol2 = np.ones((len(Acol1),1))
A = np.hstack((Acol1,Acol2))
d = psdls4dense.reshape(-1,1)
# now solve the Ax = d least-squares (LS) problem
xls = np.linalg.solve(A.T@A,A.T@d)

mls = xls[0]
bls = xls[1]

psdls = A@xls.reshape(-1,1)

# now, extract the db/octave slope:
psd1 = psdls[0] 
psd2 = psdls[-1]
freq1 = freq_list[0] 
freq2 = freq_list[-1]
psd_delta = psd2-psd1
freq_octaves = np.log10(freq2/freq1)/np.log10(2) # log2(.)=log10(.)/log10(2) <-- number of octaves!
meas_db_per_octave = psd_delta/freq_octaves

# compute error of pink noise slope relative to ideal:
# don't use this approximation of the ideal psd slope for pink noise:
# meas_psd_slope_ideal = -3.0103
# instead, compute the exact ideal psd slope for pink noise below:
meas_psd_slope_ideal = 10*np.log10(1/2)
meas_psd_slope_error = np.abs(meas_db_per_octave-meas_psd_slope_ideal)

# compute the lp-norm error of the psd relative to the least-squares psd:
meas_lpnorm_psd_error = np.linalg.norm(psdls-psdls4dense.reshape(-1,1),ord=p)

# plot the PSDs of the measured data model and the LS-estimated psd
plt.figure(figsize=(10, 6))
plt.semilogx(freq_list, psdls4dense)
plt.semilogx(freq_list, psdls, 'r--')
plt.title('power spectral density for measured data model and linear LS fit')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend(['data model psd', 'linear LS fit psd'])
plt.grid(True)
if (figsave == 1):
    if (useltspice == 0):
        plt.savefig('pinknoise_transient_testbench_audiocheck_plot4.pdf')
    else:
        plt.savefig('pinknoise_transient_testbench_ltspice_plot4.pdf')
plt.show()

if (debug == 1):
    print('p value for lp-norm:')
    print(p)
    print('score decimal places:')
    print(dec_place)
    print('db/octave of measured pink noise data:')
    print(eng_str(meas_db_per_octave[0]))
    print('ideal db/octave of pink noise:')
    print(eng_str(meas_psd_slope_ideal))
    print('measured data psd slope error:')
    print(eng_str(meas_psd_slope_error[0]))
    print('measured data lp-norm psd error:')
    print(eng_str(meas_lpnorm_psd_error))

# now use the bisection method to find a starting-point gain for ideal psd;
# first, get the linear LS psd fit value at fmin = 20 Hz, and use this value
# to assist in arriving at a set of valid starting points
gtemp = 10**(psdls[0,0]/20)

# initial lower bound and upper bound for bisection
gscale = 2.
Gl = gtemp/gscale
Gu = gtemp*gscale

# this is the bisection algorithm termination threshold
bisection_tol = 100e-12 #i.e., 1e-10

gflag = 0
kiter = 0

while (gflag == 0):
    
    kiter += 1
    
    psdglrhsvec = np.zeros([freqlistSize,1])
    psdgllhsvec = np.zeros([freqlistSize,1])
    psdgurhsvec = np.zeros([freqlistSize,1])
    psdgulhsvec = np.zeros([freqlistSize,1])
    
    for k in range(freqlistSize):
        
        f = freq_list[k]
        
        psdgl = 20*np.log10(Gl*(fmin**0.5)*(f**-0.5))
        psdgu = 20*np.log10(Gu*(fmin**0.5)*(f**-0.5))
        
        if (psdgl > psdls4dense[k,0]):
            psdglrhsvec[k,0] = psdgl - psdls4dense[k,0]
        elif (psdgl <= psdls4dense[k,0]):
            psdgllhsvec[k,0] = psdls4dense[k,0] - psdgl
        
        if (psdgu > psdls4dense[k,0]):
            psdgurhsvec[k,0] = psdgu - psdls4dense[k,0]
        elif (psdgu <= psdls4dense[k,0]):
            psdgulhsvec[k,0] = psdls4dense[k,0] - psdgu
        
    psdglrhs = np.linalg.norm(psdglrhsvec, ord=p)
    psdgllhs = np.linalg.norm(psdgllhsvec, ord=p)
    psdgurhs = np.linalg.norm(psdgurhsvec, ord=p)
    psdgulhs = np.linalg.norm(psdgulhsvec, ord=p)
    
    fgl = psdglrhs - psdgllhs
    fgu = psdgurhs - psdgulhs

    fglfgu = fgl*fgu
    
    if (fglfgu < 0):
        gflag = 1
    else:
        Gl = Gl/gscale
        Gu = Gu*gscale

GldB = 20*np.log10(Gl)
GudB = 20*np.log10(Gu)

if (debug == 1):
    print('kiter:')
    print(kiter)
    print('Gl:')
    print(eng_str(Gl))
    print('Gu:')
    print(eng_str(Gu))
    print('GldB:')
    print(eng_str(GldB))
    print('GudB:')
    print(eng_str(GudB))

psdglrhsvec = np.zeros([freqlistSize,1])
psdgllhsvec = np.zeros([freqlistSize,1])
psdgurhsvec = np.zeros([freqlistSize,1])
psdgulhsvec = np.zeros([freqlistSize,1])

for k in range(freqlistSize):
    
    f = freq_list[k]
    
    psdgl = 20*np.log10(Gl*(fmin**0.5)*(f**-0.5))
    psdgu = 20*np.log10(Gu*(fmin**0.5)*(f**-0.5))
    
    if (psdgl > psdls4dense[k,0]):
        psdglrhsvec[k,0] = psdgl - psdls4dense[k,0]
    elif (psdgl <= psdls4dense[k,0]):
        psdgllhsvec[k,0] = psdls4dense[k,0] - psdgl
    
    if (psdgu > psdls4dense[k,0]):
        psdgurhsvec[k,0] = psdgu - psdls4dense[k,0]
    elif (psdgu <= psdls4dense[k,0]):
        psdgulhsvec[k,0] = psdls4dense[k,0] - psdgu
    
psdglrhs = np.linalg.norm(psdglrhsvec, ord=p)
psdgllhs = np.linalg.norm(psdgllhsvec, ord=p)
psdgurhs = np.linalg.norm(psdgurhsvec, ord=p)
psdgulhs = np.linalg.norm(psdgulhsvec, ord=p)

fgl = psdglrhs - psdgllhs
fgu = psdgurhs - psdgulhs

ftestLU = fgl*fgu

if (debug == 1):
    print('ftestLU should be negative:')
    print(ftestLU)

u = Gu
l = Gl

kiter = 0

while Gu-Gl >= bisection_tol:
    
    kiter += 1
    
    psdgirhsvec = np.zeros([freqlistSize,1])
    psdgilhsvec = np.zeros([freqlistSize,1])
    psdglrhsvec = np.zeros([freqlistSize,1])
    psdgllhsvec = np.zeros([freqlistSize,1])
    
    # compute the next iterate to be right at the midpoint of Gl and Gu
    Gi = (Gl + Gu) / 2.
    
    for k in range(freqlistSize):
        
        f = freq_list[k]
        
        psdgi = 20*np.log10(Gi*(fmin**0.5)*(f**-0.5))
        psdgl = 20*np.log10(Gl*(fmin**0.5)*(f**-0.5))
        
        if (psdgi > psdls4dense[k,0]):
            psdgirhsvec[k,0] = psdgi - psdls4dense[k,0]
        elif (psdgi <= psdls4dense[k,0]):
            psdgilhsvec[k,0] = psdls4dense[k,0] - psdgi
        
        if (psdgl > psdls4dense[k,0]):
            psdglrhsvec[k,0] = psdgl - psdls4dense[k,0]
        elif (psdgl <= psdls4dense[k,0]):
            psdgllhsvec[k,0] = psdls4dense[k,0] - psdgl
    
    psdgirhs = np.linalg.norm(psdgirhsvec, ord=p)
    psdgilhs = np.linalg.norm(psdgilhsvec, ord=p)
    psdglrhs = np.linalg.norm(psdglrhsvec, ord=p)
    psdgllhs = np.linalg.norm(psdgllhsvec, ord=p)

    fgi = psdgirhs - psdgilhs
    fgl = psdglrhs - psdgllhs
    
    ftest = fgi*fgl
    
    if (ftest < 0):
        Gu = Gi #f(Gi) and f(Gl) differ in signs; replace Gu
    else:
        Gl = Gi #f(Gi) and f(Gl) do not differ in signs; replace Gl

G = Gi
GdB = 20*np.log10(G)

if (debug == 1):
    print('kiter:')
    print(kiter)
    print('G:')
    print(eng_str(G))
    print('GdB:')
    print(eng_str(GdB))

# generate ideal pink noise psd
psdideal = np.zeros([freqlistSize,1])

for k in range(freqlistSize):
    f = freq_list[k]
    psdideal[k]  = 20*np.log10(G*(fmin**0.5)*(f**-0.5))

if (debug == 1):
    # plot PSDs of measured data model & ideal pink noise, before optimizing
    plt.figure(figsize=(10, 6))
    plt.semilogx(freq_list, psdls4dense)
    plt.semilogx(freq_list, psdideal, 'r--')
    plt.title('PSD for measured data model and for ideal pink noise (before optimizing)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    plt.legend(['data model psd', 'ideal'])
    plt.grid(True)
    if (figsave == 1):
        if (useltspice == 0):
            plt.savefig('pinknoise_transient_testbench_audiocheck_plot5.pdf')
        else:
            plt.savefig('pinknoise_transient_testbench_ltspice_plot5.pdf')
    plt.show()

psderror = np.linalg.norm(psdideal-psdls4dense, ord=p)

# initialize optimization parameters
kiter = 0
pvar = 0.10 #start with 10% variation
pvarred = 2. #pvar reduction
opt_tol = 1e-12 #stop iterating when pvar falls below this value
psderrorbest = psderror
Gbest = G

# this is a kind of coordinate-descent algorithm without derivatives;
# we assume there is only one global minimum, that it is nearby somewhere, 
# and the response is convex, i.e., the global minimum can be attained by 
# just simply finding and following the line of descent
while pvar >= opt_tol:
    
    kiter += 1
    
    psdpvec = np.zeros([freqlistSize,1])
    psdnvec = np.zeros([freqlistSize,1])
    
    # compute the positive and negative polling iterates
    Gptemp = Gbest*(1 + pvar)
    Gntemp = Gbest*(1 - pvar)
    
    for k in range(freqlistSize):
        f = freq_list[k]
        psdpvec[k,0] = 20*np.log10(Gptemp*(fmin**0.5)*(f**-0.5))
        psdnvec[k,0] = 20*np.log10(Gntemp*(fmin**0.5)*(f**-0.5))
    
    psdperror = np.linalg.norm(psdpvec-psdls4dense, ord=p)
    psdnerror = np.linalg.norm(psdnvec-psdls4dense, ord=p)

    if (psdperror <= psdnerror)&(psdperror < psderrorbest):
        psderrorbest = psdperror
        Gbest = Gptemp
    elif (psdperror > psdnerror)&(psdnerror < psderrorbest):
        psderrorbest = psdnerror
        Gbest = Gntemp
    else:
        pvarfinal = pvar
        pvar = pvar/pvarred

if (debug == 1):
    print('kiter:')
    print(kiter)
    print('pvarfinal:')
    print(eng_str(pvarfinal))
    print('psderrorbest:')
    print(eng_str(psderrorbest))
    print('Gbest:')
    print(eng_str(Gbest))

G = Gbest
GdB = 20*np.log10(G)

psdideal = np.zeros([freqlistSize,1])
for k in range(freqlistSize):
    f = freq_list[k]
    psdideal[k]  = 20*np.log10(G*(fmin**0.5)*(f**-0.5))

# compute psd error relative to the ideal pink noise psd
psderror = np.linalg.norm(psdideal-psdls4dense, ord=p)

# compute the .tran simulated performance score (the lower, the better)
transim_performance_score = np.round(psderror*10**dec_place)/10**dec_place

if (debug == 1):
    print('G:')
    print(eng_str(G))
    print('GdB:')
    print(eng_str(GdB))
    print('psderror:')
    print(eng_str(psderror))

# plot the PSDs of the measured data model and for ideal pink noise
plt.figure(figsize=(10, 6))
plt.semilogx(freq_list, psdls4dense)
plt.semilogx(freq_list, psdideal, 'r--')
plt.title('power spectral density for measured data model and for ideal pink noise')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend(['data model psd', 'ideal'])
plt.grid(True)
if (figsave == 1):
    if (useltspice == 0):
        plt.savefig('pinknoise_transient_testbench_audiocheck_plot6.pdf')
    else:
        plt.savefig('pinknoise_transient_testbench_ltspice_plot6.pdf')
plt.show()

# display performance results:
print('useltspice:')
print(useltspice)
if (invalid_design == 1):
    print('INVALID DESIGN: positive peak and negative peak have the same sign!')
    print('Your score below may as well be infinite; resolve the issues with')
    print('your design first, then try again!')
elif (invalid_design == 0):
    print('VALID DESIGN: positive peak and negative peak have different signs')
    print('Note: just because you have a valid design, does not necessarily')
    print('mean you have an optimal design!')
print('p value for lp-norm:')
print(p)
print('psd slope error:')
print(eng_str(meas_psd_slope_error[0]))
print('lp-norm error:')
print(eng_str(meas_lpnorm_psd_error))
print('transient simulated performance score:')
print(f"{transim_performance_score:.{dec_place}f}") #f-string enforces dec_place



# results for high-quality pink noise sample audiocheck.net_pinknoise.wav:
#  useltspice:
# 0
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# psd slope error:
# 6.28788469568331e-3
# lp-norm error:
# 3.2918486068521187
# transient simulated performance score:
# 3.44



# results for pink noise output from ltspice pn_5sec_data.wav:
#  useltspice:
# 1
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# psd slope error:
# 95.95921099314131e-3
# lp-norm error:
# 32.50673457292823
# transient simulated performance score:
# 35.85
