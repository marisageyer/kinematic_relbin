#!/usr/bin/env python

import os
import sys

import math
#import random
import argparse
import logging
import subprocess
import numpy as np

import galpy
import GalDynPsr
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_context('poster')
#sns.set_style('ticks')

from shutil import copyfile
from datetime import datetime

## import binary_tools
import binary_utils as bu


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--startpar", help="provide starting parfile")
parser.add_argument("-t", "--tim", help="provide tim file")
parser.add_argument("-d", "--dist", help="provide distance to pulsar in kpc", type=float)

#### KOM, STIG and MC to loop over (low_limit, upper_limit, step_size)
parser.add_argument("-kom", "--kom", help="provide range of KOM parameters in the format: low_limit upper_limit step_size", type=float, nargs=3)
parser.add_argument("-stig", "--stig", help="provide range of STIG in the format: (low_limit, upper_limit, step_size)", type=float, nargs=3)
parser.add_argument("-mc", "--mc", help="provide range of MC in the format: (low_limit, upper_limit, step_size)", type=float, nargs=3)


parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()





if args.verbose:
    logging.basicConfig(level=logging.INFO, format='%(message)s')

logging.info('Only shown in verbose mode')


def get_Astrometry(par_file):
     with open(par_file,'r') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("RA"):
                RA = filter(None,l.split(" "))[1]                
            if l.startswith("DEC"):
                DEC = filter(None,l.split(" "))[1]                
            if l.startswith("PMRA"):
                PMRA = float(filter(None,l.split(" "))[1])                
            if l.startswith("PMDEC"):
                PMDEC = float(filter(None,l.split(" "))[1])                
            if l.startswith("A1"):
                A1 = float(filter(None,l.split(" "))[1])     
            if l.startswith("PB"):
                PB = float(filter(None,l.split(" "))[1])       
        #print("RA, DEC, PMRA, PMDEC, A1, PB:")
        #print(RA,DEC,PMRA,PMDEC, A1,PB)
        return RA, DEC, PMRA, PMDEC, A1, PB
    

def starting_par(par_file, dist):
    RA, DEC, PMRA, PMDEC, A1, PB = get_Astrometry(par_file)
    # Compute PBDOT
    pbdot = bu.Pbdot(RA,DEC,PMRA,PMDEC,dist,PB, model='C')
    ## Take the absolute value
    pbdot = abs(pbdot)
    logging.info('Computed pbdot:\t%.4e' %pbdot)
    newname = 'temp_dist%.2f.par' %(dist)
    tempfile = os.path.join(os.path.dirname(par_file),newname)
    copyfile(par_file,tempfile)
    logging.info("Created:\t\t\t%s" %tempfile)
    with open(tempfile,'a') as f:
        f.write('PBDOT            %.12e\t\t\t0\n' %pbdot)
    f.close()
    #print "Using PBDOT = %.12e" %pbdot
    return tempfile

def get_Pbdot_par(par_file):
    with open(par_file,'r') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("PBDOT"):
                PBDOT = float(filter(None, l.split("\t")[0].split(" "))[1])
    return PBDOT 

## Likelihood function: this is the chi2 computed through tempo2

def like(theta, temppar_file, tim_file, RA, DEC, PMRA, PMDEC, A1, PB, pbdot):
    #RA, DEC, PMRA, PMDEC, A1, PB = get_Astrometry(temppar_file)
    #pbdot = get_Pbdot_par(temppar_file)
    kom,stig,h3=theta
    inc_rad, inc_deg, sini = bu.inc_from_stig(stig)
    #print(inc_rad, inc_deg, sini)
    logging.info("Inclination (deg):\t%.2f" %inc_deg)
    mass_c = bu.mass_c_from_stigh3(h3, stig)
    logging.info("Mass of companion:\t%.2f" %mass_c)
    mass_p = bu.mass_p_from_massfunc(mass_c, inc_deg, A1, PB)
    logging.info("Mass of pulsar from mass function:\t%.2f" %mass_p)
    xdot = bu.xdot(inc_deg,A1,PMRA,PMDEC,kom,pbdot)
    logging.info("Proper motion contribution to XDOT:\t%.4e" %xdot)
    omdot = bu.omegadot(inc_deg,PMRA,PMDEC,kom)
    logging.info("Proper motion contribution to OMDOT:\t%.4e" %omdot)
    ## not sure if this value is correct?
    newname = 'parfile_%.1f_%.2f_%.2e.par' %(kom,stig,h3)
    newfile = os.path.join(os.path.dirname(temppar_file),newname)
    copyfile(temppar_file,newfile)
    logging.info("Created mode parfile:\t\t%s"%newfile)
    with open(newfile,'a') as f:
        f.write('KOM            %.12f\t\t\t0\n' %kom)
        f.write('KIN            %.12f\t\t\t0\n' %inc_deg)
        f.write('M2             %.12f\t\t\t0\n' %mass_c)
        f.write('OMDOT          %.12e\t\t0\n' %omdot)
        f.write('XDOT          %.12e\t\t0\n' %xdot)
    f.close()

    outname = 'tempo2_log_%.1f_%.2f_%.2e' %(kom,stig,h3)
    os_path = os.path.dirname(temppar_file)
    outpath = os.path.join(os_path,outname)
    outfile = open(outpath, 'a')    
    tempo_cmd = ['tempo2', '-f', newfile, tim_file]
    tempo_cmd_open = subprocess.Popen(tempo_cmd, shell=False, cwd='.',stdout=outfile)
    (stdoutdata, stderrdata) = tempo_cmd_open.communicate()
    logging.info("Tempo2 ouput saved to:\t\t%s" %outpath)
    chi2 = bu.chi2_from_tempo2log(outpath)
    logging.info("Removing mode parfile:\t\t%s" %newfile)
    os.remove(newfile)
    logging.info("Removing mode tempo2 log:\t%s\n" %outpath)
    os.remove(outpath)
    print("Chi-squared for mode KOM:%.2f, STIG: %.2f, INC:%.2f, H3: %.3e, Mc:%.2f, Mp:%.2f is %.2f" %(kom, stig, inc_deg, h3, mass_c, mass_p, chi2))
    print("-----------------------------------------------------------------------------------------------------")
    return chi2


def log_file():
    now = datetime.now()
    formatnow = now.strftime("%H:%M:%S")
    logfilename ="logfile_%s" %formatnow
    return logfilename


def build_grid(kom_pars, stig_pars, mc_pars, temp_par, tim_file, RA, DEC, PMRA, PMDEC, A1, PB, pbdot):
    
    KOMs = np.arange(kom_pars[0], kom_pars[1]+kom_pars[2],kom_pars[2])
    Stigs = np.arange(stig_pars[0], stig_pars[1]+stig_pars[2],stig_pars[2])
    MCs = np.arange(mc_pars[0], mc_pars[1]+mc_pars[2],mc_pars[2])

    print("Computing Chi2 grid using the following ranges:")
    print("KOM: %.3f \t to %.3f \t with step size %.3f " %(kom_pars[0], kom_pars[1],kom_pars[2]))
    print("STIG: %.3f \t to %.3f \t with step size %.3f " %(stig_pars[0], stig_pars[1],stig_pars[2]))
    print("MC: %.3f \t to %.3f \t with step size %.3f " %(mc_pars[0], mc_pars[1],mc_pars[2]))
    
    
    nmodes = len(KOMs)*len(Stigs)*len(MCs)
    print("Number of modes to solve for in grid search: %d\n" %(nmodes))
    count = 0
    logfile = log_file()
    with open(logfile,'a') as f:
        f.write("%s,%s,%s,%s\n" %("KOM", "Stig","H3", "Chi2"))
        for i in KOMs:
            for k in Stigs:
                for m in MCs:
                    h = bu.h3Shap(m,k)
                    theta=[i,k,h]
    
                    print("Mode: %d/%d\n" %(count,nmodes))
                    count+=1
            
                    chi2 = like(theta, temp_par, tim_file, RA, DEC, PMRA, PMDEC, A1, PB, pbdot)
                    f.write("%.3f,%.3f,%.4e,%.2f\n" %(i, k,h, chi2))

        f.close()

if __name__ == "__main__":
    temp_par = starting_par(args.startpar,args.dist)
    RA, DEC, PMRA, PMDEC, A1, PB = get_Astrometry(args.startpar)
    pbdot = get_Pbdot_par(temp_par)    
    build_grid(args.kom, args.stig, args.mc, temp_par, args.tim, RA, DEC, PMRA, PMDEC, A1, PB, pbdot)
