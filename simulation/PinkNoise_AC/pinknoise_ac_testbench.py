"""
Eric J. Wyers
wyers@tarleton.edu
ELEN 3445
Analog Pink Noise Generator Project

pinknoise_ac_testbench.py
"""

# this code comes with no warranty or guarantee of any kind

# this script does least-squares fitting on the psd of the pink noise 
# filter AC response and produces several error results;
# see pink_noise_shaping_filter_ac.asc
# or
# see pinknoise_ac_testbench.asc

# this script computes the ac performance score by comparing the simulated 
# ac psd to the ideal pink noise psd; the lower the score, the better

# important: when saving ltspice ac data, in the "select traces to export"
# dialog, choose format cartesian (re,im) from the dropdown menu

# also important: be sure to use the following spice directives:
# .options plotwinsize=0 numdgt=15 measdgt=15
# .ac dec 1000 20 20k

import numpy as np
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


# convenience parameters for saving plots, for selecting the file type, 
# selecting the date file, and debug results reporting and plotting
figsave = 0 #set to 1 to save figures; else 0
# png format may be more convenient for importing into your project report
figfiletype = 1 #set to 1 to save figures to pdf; else 0 for png
dataset = 0 #set to 0 to use shaping filter data; else 1 for generator data
debug = 0 #set to 1 to dump out various intermediate results; else 0

if (debug == 1):
    print('figsave:')
    print(figsave)
    print('figfiletype:')
    print(figfiletype)
    print('dataset:')
    print(dataset)

# set the value of p for the lp-norm
p = 2 #2-norm error

# set the ac performance score number of decimal places
dec_place = 2 #performance score represented to this many decimal places

# either use the modified .txt file with header removed:
# filename = 'pinknoise_ac_testbench_data_modified.txt' #header removed
# num_lines_to_remove = 0 #use this with the file with no header
# 
# or use the unmodified .txt file straight from LTSpice:
if (dataset == 0):
    filename = 'pink_noise_shaping_filter_ac_data.txt' #header not removed
elif (dataset == 1):
    filename = 'pinknoise_ac_testbench_data.txt' #header not removed
num_lines_to_remove = 1 #use this with the file with header, header has 1 line

data = np.loadtxt(filename, skiprows=num_lines_to_remove, dtype=str)

# the freq array from the first column
freq = np.array(data[:,0].astype(float)).reshape(-1,1)
# the rectangular format values from the second column
v = np.array([complex(float(s.split(',')[0]),float(s.split(',')[1])) for s in data[:,1]]).reshape(-1,1)

vmagdb = 20*np.log10(np.abs(v))
vphase = (180/np.pi)*np.angle(v)

# Create a figure and a set of subplots
fig, (ax1, ax2) = plt.subplots(2, 1) # 2 rows, 1 column

# Plot on the first subplot
ax1.semilogx(freq, vmagdb, color='black')
ax1.set_title('LTSpice AC magnitude response')
ax1.set_ylabel('|Vout/Vin| [dB]')

# Plot on the second subplot
ax2.semilogx(freq, vphase, color='black')
ax2.set_title('LTSpice AC phase response')
ax2.set_xlabel('frequency [Hz]')
ax2.set_ylabel('<Vout/Vin [deg]')

# adjust layout to prevent overlapping titles/labels
plt.tight_layout()

if (figsave == 1)&(dataset == 0):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pink_noise_shaping_filter_ac_plot1.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pink_noise_shaping_filter_ac_plot1.png')

if (figsave == 1)&(dataset == 1):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pinknoise_ac_testbench_plot1.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pinknoise_ac_testbench_plot1.png')

# display the plot
plt.show()

# optional: close the figure to free up memory if you're done with it
# plt.close()

freqvec = freq
psdvec = vmagdb #psd = power spectral density

# now, set up the linear least-squares problem
Acol1 = np.log10(freqvec).reshape(-1,1)
Acol2 = np.ones((len(Acol1),1))
A = np.hstack((Acol1,Acol2))
d = psdvec.reshape(-1,1)
# now solve the Ax = d least-squares problem
xls = np.linalg.solve(A.T@A,A.T@d)

mls = xls[0]
bls = xls[1]

psdls = A@xls.reshape(-1,1)

# now, extract the db/octave slope:
psd1 = psdls[0] 
psd2 = psdls[-1]
freq1 = freqvec[0] 
freq2 = freqvec[-1]
psd_delta = psd2-psd1
# number of octaves between f1 and f2 = log2(freq2/freq1)
freq_octaves = np.log10(freq2/freq1)/np.log10(2) #log2(.)=log10(.)/log10(2)
db_per_octave = psd_delta/freq_octaves

# compute error of pink noise slope relative to ideal:
# don't use this approximation of the ideal psd slope for pink noise:
# psd_slope_ideal = -3.0103
# instead, compute the exact ideal psd slope for pink noise below:
psd_slope_ideal = 10*np.log10(1/2)
psd_slope_error = np.abs(db_per_octave-psd_slope_ideal)

# compute the 2-norm error of the psd relative to the least-squares psd:
psd_lpnorm_error = np.linalg.norm(psdls-psdvec.reshape(-1,1),ord=p)

if (debug == 1):
    print('p value for lp-norm:')
    print(p)
    print('db/octave of pink noise data:')
    print(eng_str(db_per_octave[0]))
    print('ideal db/octave of pink noise:')
    print(eng_str(psd_slope_ideal))
    print('psd slope error:')
    print(eng_str(psd_slope_error[0]))
    print('lp-norm psd error:')
    print(eng_str(psd_lpnorm_error))

# plot the PSD and the least-squares (LS) fit
plt.figure(figsize=(10, 6))
plt.semilogx(freqvec, psdvec)
plt.semilogx(freqvec, psdls, 'r--')
plt.title('PSD response of pink noise shaping filter (AC simulation)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend(['data psd', 'LS fit psd'])
plt.grid(True)
if (figsave == 1)&(dataset == 0):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pink_noise_shaping_filter_ac_plot2.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pink_noise_shaping_filter_ac_plot2.png')
if (figsave == 1)&(dataset == 1):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pinknoise_ac_testbench_plot2.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pinknoise_ac_testbench_plot2.png')
plt.show()

# get min and max frequencies from data and size of freqvec array
fmin = freqvec[0,0]
fmax = freqvec[-1,0]
freqvecSize = len(freqvec)

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
    
    psdglrhsvec = np.zeros([freqvecSize,1])
    psdgllhsvec = np.zeros([freqvecSize,1])
    psdgurhsvec = np.zeros([freqvecSize,1])
    psdgulhsvec = np.zeros([freqvecSize,1])
    
    for k in range(freqvecSize):
        
        f = freqvec[k,0]
        
        psdgl = 20*np.log10(Gl*(fmin**0.5)*(f**-0.5))
        psdgu = 20*np.log10(Gu*(fmin**0.5)*(f**-0.5))
        
        if (psdgl > psdvec[k,0]):
            psdglrhsvec[k,0] = psdgl - psdvec[k,0]
        elif (psdgl <= psdvec[k,0]):
            psdgllhsvec[k,0] = psdvec[k,0] - psdgl
        
        if (psdgu > psdvec[k,0]):
            psdgurhsvec[k,0] = psdgu - psdvec[k,0]
        elif (psdgu <= psdvec[k,0]):
            psdgulhsvec[k,0] = psdvec[k,0] - psdgu
        
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

psdglrhsvec = np.zeros([freqvecSize,1])
psdgllhsvec = np.zeros([freqvecSize,1])
psdgurhsvec = np.zeros([freqvecSize,1])
psdgulhsvec = np.zeros([freqvecSize,1])

for k in range(freqvecSize):
    
    f = freqvec[k,0]
    
    psdgl = 20*np.log10(Gl*(fmin**0.5)*(f**-0.5))
    psdgu = 20*np.log10(Gu*(fmin**0.5)*(f**-0.5))
    
    if (psdgl > psdvec[k,0]):
        psdglrhsvec[k,0] = psdgl - psdvec[k,0]
    elif (psdgl <= psdvec[k,0]):
        psdgllhsvec[k,0] = psdvec[k,0] - psdgl
    
    if (psdgu > psdvec[k,0]):
        psdgurhsvec[k,0] = psdgu - psdvec[k,0]
    elif (psdgu <= psdvec[k,0]):
        psdgulhsvec[k,0] = psdvec[k,0] - psdgu
    
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
    
    psdgirhsvec = np.zeros([freqvecSize,1])
    psdgilhsvec = np.zeros([freqvecSize,1])
    psdglrhsvec = np.zeros([freqvecSize,1])
    psdgllhsvec = np.zeros([freqvecSize,1])
    
    # compute the next iterate to be right at the midpoint of Gl and Gu
    Gi = (Gl + Gu) / 2.
    
    for k in range(freqvecSize):
        
        f = freqvec[k,0]
        
        psdgi = 20*np.log10(Gi*(fmin**0.5)*(f**-0.5))
        psdgl = 20*np.log10(Gl*(fmin**0.5)*(f**-0.5))
        
        if (psdgi > psdvec[k,0]):
            psdgirhsvec[k,0] = psdgi - psdvec[k,0]
        elif (psdgi <= psdvec[k,0]):
            psdgilhsvec[k,0] = psdvec[k,0] - psdgi
        
        if (psdgl > psdvec[k,0]):
            psdglrhsvec[k,0] = psdgl - psdvec[k,0]
        elif (psdgl <= psdvec[k,0]):
            psdgllhsvec[k,0] = psdvec[k,0] - psdgl
    
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
psdideal = np.zeros([freqvecSize,1])

for k in range(freqvecSize):
    f = freqvec[k,0]
    psdideal[k]  = 20*np.log10(G*(fmin**0.5)*(f**-0.5))

if (debug == 1):
    # plot PSDs of simulated data and ideal pink noise, before optimizing
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqvec, psdvec)
    plt.semilogx(freqvec, psdideal, 'r--')
    plt.title('PSD for simulated data and for ideal pink noise (before optimizing)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    plt.legend(['simulated data', 'ideal'])
    plt.grid(True)
    if (figsave == 1)&(dataset == 0):
        if (figfiletype == 1):
            # save the plot as a PDF
            plt.savefig("pink_noise_shaping_filter_ac_plot3.pdf")
        else:
            # save the plot as a PNG file
            plt.savefig('pink_noise_shaping_filter_ac_plot3.png')
    if (figsave == 1)&(dataset == 1):
        if (figfiletype == 1):
            # save the plot as a PDF
            plt.savefig("pinknoise_ac_testbench_plot3.pdf")
        else:
            # save the plot as a PNG file
            plt.savefig('pinknoise_ac_testbench_plot3.png')
    plt.show()

psderror = np.linalg.norm(psdideal-psdvec, ord=p)

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
    
    psdpvec = np.zeros([freqvecSize,1])
    psdnvec = np.zeros([freqvecSize,1])
    
    # compute the positive and negative polling iterates
    Gptemp = Gbest*(1 + pvar)
    Gntemp = Gbest*(1 - pvar)
    
    for k in range(freqvecSize):
        f = freqvec[k,0]
        psdpvec[k,0] = 20*np.log10(Gptemp*(fmin**0.5)*(f**-0.5))
        psdnvec[k,0] = 20*np.log10(Gntemp*(fmin**0.5)*(f**-0.5))
    
    psdperror = np.linalg.norm(psdpvec-psdvec, ord=p)
    psdnerror = np.linalg.norm(psdnvec-psdvec, ord=p)

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

psdideal = np.zeros([freqvecSize,1])
for k in range(freqvecSize):
    f = freqvec[k,0]
    psdideal[k]  = 20*np.log10(G*(fmin**0.5)*(f**-0.5))

# compute psd error relative to the ideal pink noise psd
psderror = np.linalg.norm(psdideal-psdvec, ord=p)

# compute the .ac simulated performance score (the lower, the better)
acsim_performance_score = np.round(psderror*10**dec_place)/10**dec_place

if (debug == 1):
    print('G:')
    print(eng_str(G))
    print('GdB:')
    print(eng_str(GdB))
    print('psderror:')
    print(eng_str(psderror))

# plot the PSDs of the simulated data and for ideal pink noise
plt.figure(figsize=(10, 6))
plt.semilogx(freqvec, psdvec)
plt.semilogx(freqvec, psdideal, 'r--')
plt.title('PSD for simulated data and for ideal pink noise')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.legend(['simulated data', 'ideal'])
plt.grid(True)
if (figsave == 1)&(dataset == 0):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pink_noise_shaping_filter_ac_plot4.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pink_noise_shaping_filter_ac_plot4.png')
if (figsave == 1)&(dataset == 1):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pinknoise_ac_testbench_plot4.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pinknoise_ac_testbench_plot4.png')
plt.show()

# the slope error, lp-norm error, and ac simulation performance score for your
# shaping filter should be very similar to that listed below for my design;
# why? the results from this script are obtained with the AC testbench data, 
# and performance will only degrade once you start working in the .tran 
# simulation time domain, and will degrade even more once you start taking 
# measurements of your physical pink noise generator circuit; so if your 
# design isn't as good as it can be at this stage, it will be more difficult 
# to achieve sufficiently optimal performance in more challenging scenarios

# display performance results:
print('dataset:')
print(dataset)
print('p value for lp-norm:')
print(p)
print('psd slope error:')
print(eng_str(psd_slope_error[0]))
print('lp-norm error:')
print(eng_str(psd_lpnorm_error))
print('ac simulated performance score:')
print(f"{acsim_performance_score:.{dec_place}f}") #f-string enforces dec_place



# results for pink_noise_shaping_filter_ac.asc:
#  dataset:
# 0
# p value for lp-norm:
# 2
# psd slope error:
# 21.261516472500297e-3
# lp-norm error:
# 10.713659046720293
# ac simulated performance score:
# 11.23



# results for pinknoise_ac_testbench.asc:
#  dataset:
# 1
# p value for lp-norm:
# 2
# psd slope error:
# 21.104705915133692e-3
# lp-norm error:
# 10.695922742479176
# ac simulated performance score:
# 11.20
