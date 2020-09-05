# -*- coding: utf-8 -*-
"""
Collect and Interp data
Clean version w/o file path

Christopher Liu
8/16/2020
"""

import numpy as np
import os
from scipy.interpolate import interp1d

def extract():
    rnums = range(0,1300)
    gaugenos = [702,712,901,902,911,912]
    
    for rnum in rnums:
        outdir = '/XXX/run_%s/_output' % str(rnum).zfill(6)
        homepath = 'gauge_data/run_%s'  % str(rnum).zfill(6)
        
        if not os.path.isdir(homepath):
            os.mkdir(homepath)
        
        for gaugeno in gaugenos:
            gfile = '%s/gauge%s.txt' % (outdir, str(gaugeno).zfill(5))
            gdata = np.loadtxt(gfile)
            t     = gdata[:,1]  # seconds
            eta   = gdata[:,5]  # surface elevation in meters
            
            eta_unif, tt = unif_data(eta,t,10)
            saveloc = '%s/gauge%s.txt' % (homepath, str(gaugeno).zfill(5))
            np.savetxt(saveloc, np.vstack((tt,eta_unif)).T, delimiter=',', header="t, eta")
            
# interpolate to uniform time grid for a given sample time
def unif_data(eta,t,sample_time):
    tt = np.arange(0., t[-1], sample_time)

    gaugefcn = interp1d(t, eta, kind='linear', bounds_error=False)
    eta_unif = gaugefcn(tt)
    
    return eta_unif, tt

def main():
    if not os.path.isdir('gauge_data'):
        os.mkdir('gauge_data')
    extract()
main()




