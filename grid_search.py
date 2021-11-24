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
parser.add_argument("-h3", "--h3", help="provide range of H3 **IN UNITS 1e-8** in the format: (low_limit, upper_limit, step_size)", type=float, nargs=3)


parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()

if args.verbose:
      logging.basicConfig(level=logging.INFO, format='%(message)s')


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
        #logging.info("RA, DEC, PMRA, PMDEC, A1, PB:")
        #logging.info(RA,DEC,PMRA,PMDEC, A1,PB)
        return RA, DEC, PMRA, PMDEC, A1, PB
    

def starting_par(par_file, dist):
    RA, DEC, PMRA, PMDEC, A1, PB = get_Astrometry(par_file)
    # Compute PBDOT
    pbdot = bu.Pbdot(RA,DEC,PMRA,PMDEC,dist,PB, model='C')
    ## Take the absolute value
    pbdot = abs(pbdot)
    logging.info('Computed pbdot:\t%.4e' %pbdot)
    newname = 'temp_dist%.3f.par' %(dist)
    tempfile = os.path.join(os.path.dirname(par_file),newname)
    copyfile(par_file,tempfile)
    logging.info("Created:\t\t\t%s" %tempfile)
    with open(tempfile,'a') as f:
        f.write('PBDOT            %.12e\t\t\t0\n' %pbdot)
    f.close()
    #logging.info "Using PBDOT = %.12e" %pbdot
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
    #logging.info(inc_rad, inc_deg, sini)
    logging.info("Inclination (deg):\t%.2f" %inc_deg)
    mass_c = bu.mass_c_from_stigh3(h3, stig)
    logging.info("Mass of companion:\t%.2f" %mass_c)
    mass_p = bu.mass_p_from_massfunc(mass_c, inc_deg, A1, PB)
    logging.info("Mass of pulsar from mass function:\t%.2f" %mass_p)
    xdot = bu.xdot(inc_deg,A1,PMRA,PMDEC,kom,pbdot)
    logging.info("Proper motion contribution to XDOT:\t%.4e" %xdot)
    omdot = bu.omegadot(inc_deg,PMRA,PMDEC,kom)
    logging.info("Proper motion contribution to OMDOT:\t%.4e" %omdot)

    outname = 'tempo2_log_%.1f_%.2f_%.2e' %(kom,stig,h3)
    os_path = os.path.dirname(temppar_file)
    outpath = os.path.join(os_path,outname)
    outfile = open(outpath, 'a')    
    tempo_cmd = ['tempo2', '-f', temppar_file, tim_file, '-set', 'KOM', str(kom),'0','-set','M2',str(mass_c),'0','-set','KIN',str(inc_deg),'0', '-set','OMDOT',str(omdot), '0', '-set','XDOT',str(xdot),'0']
 #   print(tempo_cmd)
    tempo_cmd_open = subprocess.Popen(tempo_cmd, shell=False, cwd='.',stdout=outfile)
    (stdoutdata, stderrdata) = tempo_cmd_open.communicate()
    logging.info("Tempo2 ouput saved to:\t\t%s" %outpath)
    chi2 = bu.chi2_from_tempo2log(outpath)
    logging.info("Removing mode tempo2 log:\t%s\n" %outpath)
    os.remove(outpath)
    logging.info("Chi-squared for mode KOM: %.2f, STIG: %.2f, INC: %.2f, H3: %.3e, Mc: %.2f, Mp: %.2f is %.2f" %(kom, stig, inc_deg, h3, mass_c, mass_p, chi2[0]))
    logging.info("Reduced chi-squared for mode KOM: %.2f, STIG: %.2f, INC: %.2f, H3: %.3e, Mc: %.2f, Mp: %.2f is %.4f" %(kom, stig, inc_deg, h3, mass_c, mass_p, chi2[1]))
    logging.info("-----------------------------------------------------------------------------------------------------")
    return chi2


def log_file():
    now = datetime.now()
    formatnow = now.strftime("%d-%m-%Y_%H-%M")
    logfilename ="logfile_%s" %formatnow
    return logfilename


if args.mc:
    def build_grid(kom_pars, stig_pars, mc_pars, temp_par, tim_file, RA, DEC, PMRA, PMDEC, A1, PB, pbdot):
        
        KOMs = np.arange(kom_pars[0], kom_pars[1]+kom_pars[2],kom_pars[2])
        Stigs = np.arange(stig_pars[0], stig_pars[1]+stig_pars[2],stig_pars[2])
        MCs = np.arange(mc_pars[0], mc_pars[1]+mc_pars[2],mc_pars[2])
    
        logging.info("Computing Chi2 grid using the following ranges:")
        logging.info("KOM: %.3f \t to %.3f \t with step size %.3f " %(kom_pars[0], kom_pars[1],kom_pars[2]))
        logging.info("STIG: %.3f \t to %.3f \t with step size %.3f " %(stig_pars[0], stig_pars[1],stig_pars[2]))
        logging.info("MC: %.3f \t to %.3f \t with step size %.3f " %(mc_pars[0], mc_pars[1],mc_pars[2]))
        
        
        nmodes = len(KOMs)*len(Stigs)*len(MCs)
        ninner = len(Stigs)*len(MCs)
        logging.info("Number of modes to solve for in grid search: %d\n" %(nmodes))
        count = 0
        logfile = log_file()
        
        all_out = np.zeros([nmodes,5])
        with open(logfile,'a') as f:
            f.write("%s,%s,%s,%s,%s\n" %("KOM", "Stig","H3*1e8", "Chi2","RedChi2"))
            for ii,i in enumerate(KOMs):
                for k in Stigs:
                    for m in MCs:
                        h = bu.h3Shap(m,k)
                        theta=[i,k,h]
        
                        logging.info("Mode: %d/%d\n" %(count+1,nmodes))
                
                        chi2 = like(theta, temp_par, tim_file, RA, DEC, PMRA, PMDEC, A1, PB, pbdot)
                        #f.write("%.4f,%.4f,%.4e,%.3f,%.3f\n" %(i, k,h, chi2[0], chi2[1]))
                        all_out[count,:] = [i,k,h*1e8,chi2[0],chi2[1]]
                        count+=1
                np.savetxt(f,all_out[ii*ninner:(ii+1)*ninner],fmt="%.4f", delimiter=",", newline="\n")        
            f.close()

elif args.h3:
    def build_grid(kom_pars, stig_pars, h3_pars, temp_par, tim_file, RA, DEC, PMRA, PMDEC, A1, PB, pbdot):
         
        KOMs = np.arange(kom_pars[0], kom_pars[1]+kom_pars[2],kom_pars[2])
        Stigs = np.arange(stig_pars[0], stig_pars[1]+stig_pars[2],stig_pars[2])
        H3s = 1e-8*np.arange(h3_pars[0], h3_pars[1]+h3_pars[2],h3_pars[2])
    
        logging.info("Computing Chi2 grid using the following ranges:")
        logging.info("KOM: %.3f \t to %.3f \t with step size %.3f " %(kom_pars[0], kom_pars[1],kom_pars[2]))
        logging.info("STIG: %.3f \t to %.3f \t with step size %.3f " %(stig_pars[0], stig_pars[1],stig_pars[2]))
        logging.info("H3: %.3f \t to %.3f \t with step size %.3f " %(h3_pars[0], h3_pars[1],h3_pars[2]))
        
        nmodes = len(KOMs)*len(Stigs)*len(H3s)
        ninner = len(Stigs)*len(H3s)
        logging.info("Number of modes to solve for in grid search: %d\n" %(nmodes))
        count = 0
        logfile = log_file()
       
        all_out=np.zeros([nmodes,5])
        with open(logfile,'a') as f:
            f.write("%s,%s,%s,%s,%s\n" %("KOM", "Stig","H3*1e8", "Chi2","RedChi2"))
            for ii,i in enumerate(KOMs):
                for k in Stigs:
                    for h in H3s:
                        theta=[i,k,h]
                        logging.info("Mode: %d/%d\n" %(count+1,nmodes))
                
                        chi2 = like(theta, temp_par, tim_file, RA, DEC, PMRA, PMDEC, A1, PB, pbdot)
                        #str_to_write ="%.4f,%.4f,%.4e,%.3f,%.3f\n" %(i, k,h, chi2[0], chi2[1])
                        all_out[count,:] = [i,k,h*1e8,chi2[0],chi2[1]]
                        count+=1

                np.savetxt(f,all_out[ii*ninner:(ii+1)*ninner],fmt="%.4f", delimiter=",", newline="\n")        
            f.close()

if __name__ == "__main__":
    temp_par = starting_par(args.startpar,args.dist)
    RA, DEC, PMRA, PMDEC, A1, PB = get_Astrometry(args.startpar)
    pbdot = get_Pbdot_par(temp_par)


    if all(i is None for i in [args.mc, args.h3]):
        logging.info("Provide at least one of MC or H3 along with KOM and Stig")
    if args.mc:    
        build_grid(args.kom, args.stig, args.mc, temp_par, args.tim, RA, DEC, PMRA, PMDEC, A1, PB, pbdot)
    if args.h3:    
        build_grid(args.kom, args.stig, args.h3, temp_par, args.tim, RA, DEC, PMRA, PMDEC, A1, PB, pbdot)
