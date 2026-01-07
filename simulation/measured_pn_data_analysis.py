"""
Eric J. Wyers
wyers@tarleton.edu
ELEN 3445
Analog Pink Noise Generator Project

measured_pn_data_analysis.py
"""

# this code comes with no warranty or guarantee of any kind

# this script analyzes 50-ms duration .txt data files to determine key
# performance details and produce a score; data should be sampled at a 
# rate of 50 kHz, which corresponds to a sample period of 20 us, and should
# be able to adequately capture the voltage range of the measured quantity;
# the FFT is used to compute the PSD

# this script computes the measured performance project score by comparing the
# measured psd to the ideal pink noise psd; the lower the score, the better

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# convenience parameters for saving plots to pdf, selecting .txt file, 
# model selection, and debug results reporting and plotting
figsave = 0 #set to 1 to save figures; else 0
fileselect = 0 #set value to correspond to data file(s) below; indexed to 0
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
    print('fileselect:')
    print(fileselect)
    print('model_select:')
    print(model_select)

# set the value of p for the lp-norm
p = 2 #2-norm error

# set the measured performance project score number of decimal places
dec_place = 2 #performance score represented to this many decimal places

# import measured data from .txt file
if (fileselect == 0):
    filename = 'measured_data_sample.txt' #note: header not removed
elif (fileselect == 1):
    filename = 'measured_pn_data1.txt' #note: header not removed
elif (fileselect == 2):
    filename = 'measured_pn_data2.txt' #note: header not removed
elif (fileselect == 3):
    filename = 'measured_pn_data3.txt' #note: header not removed
elif (fileselect == 4):
    filename = 'measured_pn_data4.txt' #note: header not removed
elif (fileselect == 5):
    filename = 'measured_pn_data5.txt' #note: header not removed
elif (fileselect == 6):
    filename = 'measured_pn_data6.txt' #note: header not removed
elif (fileselect == 7):
    filename = 'measured_pn_data7.txt' #note: header not removed
# elif (fileselect == 8): #add more, if needed, or comment out unneeded ones
    # filename = 'measured_pn_data8.txt' #note: header not removed
num_lines_to_remove = 5 #header has 5 lines

# now, extract data from .txt file
column1 = []
column2 = []
column2temp = []
column3 = []
k = -1

with open(filename, 'r') as file:
    for line in file:
        k += 1
        if (k >= num_lines_to_remove):
            data = line.strip().split() # assuming space-separated values
            if len(data) >= 2: # ensure there are enough columns
                column1_value = data[0] #column1 is the date, not needed
                column2_value = data[1] #column2 is the time in h:m:s format
                column3_value = float(data[2]) #column3 is the measurement
                column1.append(column1_value)
                column2.append(column2_value)
                column3.append(column3_value)

def get_total_seconds(time_str):
  """
  Converts a time string in H:M:S format to total seconds.
  Handles cases where components might not be zero-padded.
  """
  try:
    parts = time_str.split(':')
    if len(parts) == 3:  # H:M:S
      h, m, s = [float(p) for p in parts]
      return h * 3600 + m * 60 + s
    elif len(parts) == 2:  # M:S (assuming 0 hours)
      m, s = [float(p) for p in parts]
      return m * 60 + s
    elif len(parts) == 1:  # S (assuming 0 hours and 0 minutes)
      s = float(parts[0])
      return s
    else:
      raise ValueError("Invalid time format")
  except ValueError as e:
    print(f"Error converting '{time_str}': {e}")
    return None  # or raise the exception again, depending on your error-handling needs

for item in column2:
  time_string = item.strip()  # remove leading/trailing whitespace
  total_seconds = get_total_seconds(time_string)
  if total_seconds is not None:
    column2temp.append(total_seconds)

# column2 is the time samples; convert from list to numpy array
column2 = np.array(column2temp)

# noise_signal_meas is the measured output voltage data; convert from list to numpy array
noise_signal_meas = np.array(column3)

sampling_rate_meas = 1/(column2[1] - column2[0])
sampling_rate_meas = np.round(sampling_rate_meas).astype(int) #round to nearest integer
T_meas = 1/sampling_rate_meas
tmin_meas = 0.
tmax_meas = (len(noise_signal_meas)-1)*T_meas
tvec_meas = np.arange(tmin_meas, tmax_meas+T_meas, T_meas)

plt.figure(figsize=(10, 6))
plt.plot(tvec_meas, noise_signal_meas)
plt.title('measured output voltage')
plt.xlabel('time [s]')
plt.ylabel('voltage [V]')
plt.grid(True)
if (figsave == 1):
    if (fileselect == 0):
        plt.savefig('measured_pn_analysis_file0_plot0.pdf')
    elif (fileselect == 1):
        plt.savefig('measured_pn_analysis_file1_plot0.pdf')
    elif (fileselect == 2):
        plt.savefig('measured_pn_analysis_file2_plot0.pdf')
    elif (fileselect == 3):
        plt.savefig('measured_pn_analysis_file3_plot0.pdf')
    elif (fileselect == 4):
        plt.savefig('measured_pn_analysis_file4_plot0.pdf')
    elif (fileselect == 5):
        plt.savefig('measured_pn_analysis_file5_plot0.pdf')
    elif (fileselect == 6):
        plt.savefig('measured_pn_analysis_file6_plot0.pdf')
    elif (fileselect == 7):
        plt.savefig('measured_pn_analysis_file7_plot0.pdf')
plt.show()

if (debug == 1):
    print('sample period [s]:')
    print(eng_str(T_meas))
    print('sample frequency [Hz]:')
    print(eng_str(sampling_rate_meas))
    print('last time sample [s]:')
    print(eng_str(tvec_meas[-1]))
    print('measured noise signal mean [V]:')
    print(eng_str(np.mean(noise_signal_meas)))
    print('measured noise signal max [V]:')
    print(eng_str(np.max(noise_signal_meas)))
    print('measured noise signal min [V]:')
    print(eng_str(np.min(noise_signal_meas)))
    print('measured noise signal peak-to-peak, pp [V]:')
    print(eng_str(np.max(noise_signal_meas)-np.min(noise_signal_meas)))

# plot the histogram of the noise signal
plt.hist(noise_signal_meas, bins=30, edgecolor='black')
plt.xlabel("value")
plt.ylabel("frequency")
plt.title("histogram of measured pink noise signal")
if (figsave == 1):
    if (fileselect == 0):
        plt.savefig('measured_pn_analysis_file0_plot1.pdf')
    elif (fileselect == 1):
        plt.savefig('measured_pn_analysis_file1_plot1.pdf')
    elif (fileselect == 2):
        plt.savefig('measured_pn_analysis_file2_plot1.pdf')
    elif (fileselect == 3):
        plt.savefig('measured_pn_analysis_file3_plot1.pdf')
    elif (fileselect == 4):
        plt.savefig('measured_pn_analysis_file4_plot1.pdf')
    elif (fileselect == 5):
        plt.savefig('measured_pn_analysis_file5_plot1.pdf')
    elif (fileselect == 6):
        plt.savefig('measured_pn_analysis_file6_plot1.pdf')
    elif (fileselect == 7):
        plt.savefig('measured_pn_analysis_file7_plot1.pdf')
plt.show()

maxmeasdata = np.max(noise_signal_meas)
minmeasdata = np.min(noise_signal_meas)

# check if design is valid or not based on the signs of the voltage peaks
if (maxmeasdata/minmeasdata > 0):
    invalid_design = 1 #this means that max output and min output have the same signs
else:
    invalid_design = 0 #this means that max output is positive and min output is negative

# now conpute the psd of data via the fft
N = len(noise_signal_meas)
Dfft = np.fft.rfft(noise_signal_meas,N) #for N even, Dfft should be N/2 + 1 in length
NDfft = len(Dfft)
Ffftmin = 0.
Ffftmax = 1.
Ffftstep = (Ffftmax - Ffftmin)/(NDfft - 1)
frequencies = (sampling_rate_meas/2)*np.arange(Ffftmin, Ffftmax+Ffftstep, Ffftstep)

psd = (1/(sampling_rate_meas*N))*np.abs(Dfft)**2
psd[1:-2] = 2*psd[1:-2]

# save a copy of the psd outputs above for plot2 (plots are indexed to 0)
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

# save a copy of these arrays for plot2 (plots are indexed to 0)
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
    if (fileselect == 0):
        plt.savefig('measured_pn_analysis_file0_plot2.pdf')
    elif (fileselect == 1):
        plt.savefig('measured_pn_analysis_file1_plot2.pdf')
    elif (fileselect == 2):
        plt.savefig('measured_pn_analysis_file2_plot2.pdf')
    elif (fileselect == 3):
        plt.savefig('measured_pn_analysis_file3_plot2.pdf')
    elif (fileselect == 4):
        plt.savefig('measured_pn_analysis_file4_plot2.pdf')
    elif (fileselect == 5):
        plt.savefig('measured_pn_analysis_file5_plot2.pdf')
    elif (fileselect == 6):
        plt.savefig('measured_pn_analysis_file6_plot2.pdf')
    elif (fileselect == 7):
        plt.savefig('measured_pn_analysis_file7_plot2.pdf')
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
    if (fileselect == 0):
        plt.savefig('measured_pn_analysis_file0_plot3.pdf')
    elif (fileselect == 1):
        plt.savefig('measured_pn_analysis_file1_plot3.pdf')
    elif (fileselect == 2):
        plt.savefig('measured_pn_analysis_file2_plot3.pdf')
    elif (fileselect == 3):
        plt.savefig('measured_pn_analysis_file3_plot3.pdf')
    elif (fileselect == 4):
        plt.savefig('measured_pn_analysis_file4_plot3.pdf')
    elif (fileselect == 5):
        plt.savefig('measured_pn_analysis_file5_plot3.pdf')
    elif (fileselect == 6):
        plt.savefig('measured_pn_analysis_file6_plot3.pdf')
    elif (fileselect == 7):
        plt.savefig('measured_pn_analysis_file7_plot3.pdf')
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
        if (fileselect == 0):
            plt.savefig('measured_pn_analysis_file0_plot4.pdf')
        elif (fileselect == 1):
            plt.savefig('measured_pn_analysis_file1_plot4.pdf')
        elif (fileselect == 2):
            plt.savefig('measured_pn_analysis_file2_plot4.pdf')
        elif (fileselect == 3):
            plt.savefig('measured_pn_analysis_file3_plot4.pdf')
        elif (fileselect == 4):
            plt.savefig('measured_pn_analysis_file4_plot4.pdf')
        elif (fileselect == 5):
            plt.savefig('measured_pn_analysis_file5_plot4.pdf')
        elif (fileselect == 6):
            plt.savefig('measured_pn_analysis_file6_plot4.pdf')
        elif (fileselect == 7):
            plt.savefig('measured_pn_analysis_file7_plot4.pdf')
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

# compute the measured performance score (the lower, the better)
project_score = np.round(psderror*10**dec_place)/10**dec_place

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
    if (fileselect == 0):
        plt.savefig('measured_pn_analysis_file0_plot5.pdf')
    elif (fileselect == 1):
        plt.savefig('measured_pn_analysis_file1_plot5.pdf')
    elif (fileselect == 2):
        plt.savefig('measured_pn_analysis_file2_plot5.pdf')
    elif (fileselect == 3):
        plt.savefig('measured_pn_analysis_file3_plot5.pdf')
    elif (fileselect == 4):
        plt.savefig('measured_pn_analysis_file4_plot5.pdf')
    elif (fileselect == 5):
        plt.savefig('measured_pn_analysis_file5_plot5.pdf')
    elif (fileselect == 6):
        plt.savefig('measured_pn_analysis_file6_plot5.pdf')
    elif (fileselect == 7):
        plt.savefig('measured_pn_analysis_file7_plot5.pdf')
plt.show()

# display performance results:
print('fileselect:')
print(fileselect)
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
print('measured data psd slope error:')
print(eng_str(meas_psd_slope_error[0]))
print('measured data lp-norm psd error:')
print(eng_str(meas_lpnorm_psd_error))
print('measured project score (the lower, the better):')
print(f"{project_score:.{dec_place}f}") #f-string enforces dec_place



# results for measured_data_sample.txt data file:
#  fileselect:
# 0
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 3.1992282688535587e0
# measured data lp-norm psd error:
# 50.59281271242035e0
# measured project score (the lower, the better):
# 506.81



# results for measured_pn_data1.txt data file:
#  fileselect:
# 1
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 2.455733183605191e0
# measured data lp-norm psd error:
# 193.76044636187459e0
# measured project score (the lower, the better):
# 432.87



# results for measured_pn_data2.txt data file:
#  fileselect:
# 2
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 2.2634631445910482e0
# measured data lp-norm psd error:
# 177.80793779318333e0
# measured project score (the lower, the better):
# 398.63



# results for measured_pn_data3.txt data file:
#  fileselect:
# 3
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 639.603577115595e-3
# measured data lp-norm psd error:
# 105.56238015907637e0
# measured project score (the lower, the better):
# 145.97



# results for measured_pn_data4.txt data file:
#  fileselect:
# 4
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 581.712118810597e-3
# measured data lp-norm psd error:
# 145.37421181875504e0
# measured project score (the lower, the better):
# 171.88



# results for measured_pn_data5.txt data file:
#  fileselect:
# 5
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 31.8574343383764e-3
# measured data lp-norm psd error:
# 210.4691026973646e0
# measured project score (the lower, the better):
# 210.53



# results for measured_pn_data6.txt data file:
#  fileselect:
# 6
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 973.4676976273562e-3
# measured data lp-norm psd error:
# 100.41540173760488e0
# measured project score (the lower, the better):
# 183.38



# results for measured_pn_data7.txt data file:
#  fileselect:
# 7
# VALID DESIGN: positive peak and negative peak have different signs
# Note: just because you have a valid design, does not necessarily
# mean you have an optimal design!
# p value for lp-norm:
# 2
# measured data psd slope error:
# 65.34537550917507e-3
# measured data lp-norm psd error:
# 147.49960363058707e0
# measured project score (the lower, the better):
# 147.86
