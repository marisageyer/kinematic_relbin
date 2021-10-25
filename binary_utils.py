#!/usr/bin/env python

import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from sympy import sin,cos

from astropy import units as u
from astropy.coordinates import SkyCoord

import galpy
import GalDynPsr

## Setting up equations and parameters


def Pbdot(RA,DEC,PMRA,PMDEC,dist,Pb,model='A'):
    ## Contribution to Pbdot due to proper motion
    ## This requires a galactic model for which we are using GalDynPsr (https://github.com/pathakdhruv/GalDynPsr)
    ## In particular model A (https://arxiv.org/pdf/1712.06590.pdf)
    ## Can also add error propagation. Leaving that for now as we will be running emcee across parameter space. 
    """RA  -- hour angle as string h:m:s.s
       DEC -- as a string in deg:m:s
       PMRA, PMDEC - proper motion of pulsar in RA or DEC in mas/yr
       dist -- distance to pulsar in kpc
       Pb -- orbital period -- the chosen unit for Pb will determine the unit of Pbdot

       """
    
    ldeg, bdeg = radec_to_glgb(RA,DEC)
    # ldeg = Galactic logitude in degrees, 
    # bdeg = Galactic latitude in degrees, 
    # dkpc = distance to pulsar in kpc, 

    # sigl = error in ldeg, 
    # sigb = error in bdeg, 
    # sigd = error in dkpc
    
    #Add errors to function later if want to
    #sigl = 0.00005355054359974704
    #sigb = 0.00067411170940802633
    #sigd = 0.01
    
    sigl=0
    sigb=0
    sigd=0
    sigmua=0
    sigmud=0


    ############# Extract important parameters say values of Rp (in kpc) and z (in kpc) and errors in them ##########
    
    dkpc = dist
    
    Rpkpc = GalDynPsr.read_parameters.Rpkpc(ldeg, sigl, bdeg, sigb, dkpc, sigd)
    zkpc = GalDynPsr.read_parameters.z(ldeg, sigl, bdeg, sigb, dkpc, sigd)
    ErrRp = GalDynPsr.read_parameters.ErrRp(ldeg, sigl, bdeg, sigb, dkpc, sigd)
    Errz = GalDynPsr.read_parameters.Errz(ldeg, sigl, bdeg, sigb, dkpc, sigd)

    ################# Compute excess Shklovskii term. Here Exshk() calculates the dynamical contribution (or the excess term) due to Shklovskii effect and errShk() calculates the error in that #################################

    ##  sigmua = error in mu_alpha, and sigmud = error in mu_delta

    mu_alpha = PMRA
    mu_delta = PMDEC

    #Add errors to function if want to
    #sigmua = 0.02365429289594718869
    #sigmud = 0.04209234992356018085

    ExcessSh = GalDynPsr.Shk.Exshk(dkpc, sigd, mu_alpha, sigmua, mu_delta, sigmud)
    #sigmaExcessSh = GalDynPsr.Shk.errShk(dkpc, sigd, mu_alpha, sigmua, mu_delta, sigmud)


    print("Using Model %s from GalDynPsr in computation of Pbdot" %model)
    
    # calculate the planar contribution to the excess term
    #Apl = GalDynPsr.modelB.Expl(ldeg, sigl, bdeg, sigb, dkpc, sigd) 
    exec('Apl=GalDynPsr.model%s.Expl(ldeg, 0, bdeg, 0, dist, 0)' %model)
    # calculate the perpendicular contribution to the excess term
    #Az = GalDynPsr.modelB.Exz(ldeg, sigl, bdeg, sigb, dkpc, sigd) 
    exec('Az=GalDynPsr.model%s.Exz(ldeg, 0, bdeg, 0, dist, 0)' %model)
    
    # calculate the error in the planar contribution to the excess term
    #Aplsigma = GalDynPsr.modelA.Errpl(ldeg, sigl, bdeg, sigb, dkpc, sigd)
    #calculate the error in the perpendicular contribution to the excess term
    #Azsigma = GalDynPsr.modelA.Errz(ldeg, sigl, bdeg, sigb, dkpc, sigd) 

    totalA = np.abs(Apl) + np.abs(Az)
    #print("TotalA is sum of abs(Apl) and abs(Az):", Apl, Az)
    #print("TotalA:", totalA)
    #print np.abs(Apl)+np.abs(Az)
    
    # assuming no correlation between excepp_pl and excess_z    
    #SigmatotalA = math.sqrt(Aplsigma*Aplsigma+Azsigma*Azsigma) 

    Pb_sec = Pb*24*3600
    pbdot_out = 0 - (ExcessSh + totalA)*Pb_sec 
    ## 0 for all the other terms in e.g. eq.8.72 and then this dominant term subtracted from that
    
    #return ExcessSh, totalA, 
    print("Note, at the moment a positive PBDOT value is used in the grid searches to follow.")
    return pbdot_out



## Redoing GalDynPsr by computing equations in Graikou
def Shklovskii(PMRA,PMDEC,dist):
    prop_mag = np.sqrt(PMRA**2 + PMDEC**2)
    shlov_out  = 2.43e-21*prop_mag**2*dist
    return shlov_out
    

def xdot(inc, xp, PMRA, PMDEC, omega_asc, pbdot):
    ## Contribution to xdot due to proper motion in s/s (unitless)
    ## inc - inclination angle in degrees
    ## xp - projected semi-major axis (lightsecs)
    ## omega_asc - longtitude of the ascending node in degrees (KOM in tempo2)
    ## PMRA, PMDEC - proper motion of pulsar in RA or DEC in mas/yr
    
    omega_rad = np.deg2rad(omega_asc)
    inc_rad = np.deg2rad(inc)
    
    xdot_t1 = 1.54*1e-16*xp*(1.0/np.tan(inc_rad))*(-PMRA*np.sin(omega_rad) + PMDEC*np.cos(omega_rad))
    #xdot_t2 = pbdot
    #xdot_out = xdot_t1 + xdot_t2 ## is this right??
    xdot_out = xdot_t1
    return xdot_out

def omegadot(inc, PMRA, PMDEC, omega_asc):
    ## Contribution to omegadot due to proper motion in deg/yr
    ## inc - inclination angle in degrees
    ## omega_asc - longtitude of the ascending node in degrees (KOM in tempo2)
    ## PMRA, PMDEC - proper motion of pulsar in RA or DEC in mas/yr   
    
    omega_rad = np.deg2rad(omega_asc)
    inc_rad = np.deg2rad(inc)
    
    mastodeg = 1.0/(3.6e6) 
    
    omegadot_out = mastodeg*(1.0/np.sin(inc_rad))*(PMRA*np.cos(omega_rad) + PMDEC*np.sin(omega_rad))
    return omegadot_out


def massfunction(mp,mc,inc):
    """
    mc - companion mass (units of solar mass)
    mp - pulsar mass (units of solar mass)
    inc - orbital inclination (degrees)"""
    inc_rad = np.deg2rad(inc)
    massfunc_out = (mc*np.sin(inc_rad))**3/(mp + mc)**2
    return massfunc_out


## Coordinate transform from RA, DEC to GL and GB

def radec_to_glgb(ra,dec):
    """ra  -- hour angle as string h:m:s.s
    dec -- as a string in deg:m:s"""
    radec = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    gal = radec.galactic
    gl = gal.l.deg
    gb = gal.b.deg
    return gl, gb


## Shapiro parameters

Tsun = 4.9254909476412675*1e-6 ## mass of the Sun in units of time (G*M_sun/c^3)

def rShap(mc):
    ## mc - mass of companion (units??) 
    Tsun = 4.9254909476412675*1e-6 ## mass of the Sun in units of time (sec)(G*M_sun/c^3)
    rShap_out = Tsun*mc
    return rShap_out

def sShap(inc):
    ## inc - orbital inclination in degrees
    inc_rad = np.deg2rad(inc)
    sShap_out = np.sin(inc_rad)
    return sShap_out

## Reparametrised as stig and h3, these parameters have smaller correlations than r and s, 
## and hence provide a better description of mc and inc (especially for weak Shapiro detections)

def stigShap(inc):
    ## stig: orthometeric ratio of Shapiro delay
    ## inc - orbital inclination in degrees
    inc_rad = np.deg2rad(inc)
    stigShap_out = np.sin(inc_rad)/(1+np.abs(np.cos(inc_rad)))
    return stigShap_out

def h3Shap(mc,stig):
    ## h3: orthometric amplitude of Shapiro delay
    Tsun = 4.9254909476412675*1e-6 ## mass of the Sun in units of time (sec)(G*M_sun/c^3)
    h3Shap_out = Tsun*mc*stig**3
    return h3Shap_out

## Inverted functions to obtain inc, m1 and m2 from h3 and stig.

def inc_from_stig(stig):
    # Note currently assumes a single solution (as for 0<i<90)
    i = Symbol('i')
    sol =  solve(stig - sin(i)/(1+cos(i)), i)
    inc_rad = (float(sol[0]))
    inc_deg = np.rad2deg(inc_rad)
    sini = np.sin(inc_rad)
    #print('RAD','DEG','SINI')
    return inc_rad, inc_deg, sini

def mass_c_from_stigh3(h3,stig):
    Tsun = 4.9254909476412675*1e-6 ## mass of the Sun in units of time (sec)(G*M_sun/c^3)
    mc_out = h3/(Tsun*stig**3)
    return mc_out

def mass_p_from_massfunc(mc, inc, xp, Pb):
    """
    mc - companion mass (units of solar mass)
    inc - orbital inclination (degrees)
    xp - projected semi-major axis (lightsecs)
    Pb - orbital period (days)
    Returns mass of pulsar in units of solar mass
    """
    #Tsun = 4.9254909476412675*1e-6 ## mass of the Sun in units of time (sec)(G*M_sun/c^3)
    inc_rad = np.deg2rad(inc)
    Pb_hr = Pb*24
    #Pb_sec = Pb_hr*3600
    #RHS0 = 4*(np.pi)**2*(xp**3)/(Tsun*Pb_sec**2)
    RHS = 0.618*xp**3/Pb_hr**2
    #print "RHS",RHS, RHS0
    mp_out = np.sqrt((mc*np.sin(inc_rad))**3/RHS) - mc
    return mp_out

def chi2_from_tempo2log(tempo2log):
    f = open(tempo2log)
    lines = f.readlines()
    ## Find the Fit Chi line:
    idx_Chi = np.where([lines[i].startswith("Fit Chi") for i in range(len(lines))])[0][0]
    Chi_line = lines[idx_Chi].split("\t")[0]
    Chi = float(Chi_line.split("= ")[-1])
    red_Chi_line = lines[idx_Chi].split("\t")[1]
    red_Chi = float(red_Chi_line.split("= ")[-1])
    return [Chi,red_Chi]
