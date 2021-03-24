
"""
Extract Nankai data and output .npy file
Authors: Chris Liu, Donsub Rim
"""

import numpy as np
import os

outdir = r'C:\Users\Chris\Desktop\Tsunami Personal\TohokuU_Data\Nankai_trunc_multi' # Outer directory containing the raw data
magnitudes = [7.6, 7.9, 8.2, 8.5, 8.8]
#runs = [1, 2, 3 ,4, 5] # Starts from run#1
points = ['2_05',
         '2_06',
         '2_07',
         '2_13',
         '2_14',
         '2_15',
         '2_17',
         '2_18',
         '3_23',
         '3_74',
         '5_29',
         '5_38']
 
npts = 6*60*12 #change if interp, currently for 5 second grid over 6 hours of data.
nruns_all = 1564
eta_all = np.zeros((nruns_all, len(points), npts))
t = np.zeros((nruns_all, len(points), npts))

smooth = False

# gaussian kernel
sigma = 0.2
x = np.arange(-3*sigma, 3*sigma, 0.2)
gaussian = np.exp(-(x/sigma)**2/2)

runtotal = 0 #shifted all the runs so it starts from 0
for mag in magnitudes:
    magdir = os.path.join(outdir,'JNan_%s' % str(mag))
    
    tot_runs = len(os.listdir(magdir))
    runs = np.arange(1,tot_runs+1)
    
    for runno in runs:
        rundir = os.path.join(magdir,'JNan_{:1.1f}_{:03d}'.format(mag, runno))
        
        for n, point in enumerate(points):
            file = os.path.join(rundir, 'pnt{:s}.asc'.format(point))        
            
            if os.path.isfile(file):
                data = np.genfromtxt(file, delimiter=',', skip_header=1)
                
                if smooth:
                    gaussian /= np.sum(gaussian)
                    t[runtotal,n,:] = data[:,0]
                    eta_all[runtotal,n,:] = np.convolve(data[:,1], gaussian, mode='same')
                elif not smooth:
                    t[runtotal,n,:] = data[:,0]
                    eta_all[runtotal,n,:] = data[:,1]
                    
        runtotal += 1

# Save files
savedir = r'C:\Users\Chris\Desktop\Tsunami Personal\TohokuU_Data\npy'
if not os.path.isdir(savedir):
    os.mkdir(savedir)


np.save(os.path.join(savedir,'nankai_eta.npy'), eta_all)
np.save(os.path.join(savedir,'nankai_time.npy'), t)