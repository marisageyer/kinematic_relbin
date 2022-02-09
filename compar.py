#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Compare the parameters, those fitted and the jumps included between two parameter files")
parser.add_argument("-p", "--parfiles", help="provide two parfiles to compare, separated by spaces e.g. -p par1.par par2.par", nargs=2)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

## functions

def par_to_tempfiles(par1,par2):
    temp1 = open('temp1.csv', 'w')
    temp2 = open('temp2.csv', 'w')
    
    with open(par1, 'r') as f1:
        lines = f1.readlines()
        for l in lines:
            space2tab = "\t".join(l.split())+"\n"
            temp1.write(space2tab) 
    temp1.close()
    
    with open(par2, 'r') as f2:
        lines = f2.readlines()
        for l in lines:
            space2tab = "\t".join(l.split())+"\n"
            temp2.write(space2tab) 
    temp2.close()
    #print("Created temporary files %s and %s - which will be deleted upon exit" %(temp1.name, temp2.name))
    
def file_to_frame(file1,file2):
    names=['parname', 'parvalue', 'fitflag', 'uncertainty', 'extra']
    df1 = pd.read_csv(file1, delimiter='\t',header = None,names=names,comment='#')
    df2 = pd.read_csv(file2, delimiter='\t',header = None,names=names,comment='#')
    
    return df1, df2

def check_psrnames(df1,df2):
    psr1 = df1[df1.parname == 'PSRJ'].parvalue.values[0]
    psr2 = df2[df2.parname == 'PSRJ'].parvalue.values[0]
    if psr1 != psr2:
        print(f'---- NOTE: The par files are for different pulsars: {psr1} and {psr2} -----')


def fit_par(df):
    ## remove the jumps from dataframe, will be analysed separately
    df = df[~df.parname.str.contains('JUMP')]
    df_fit = df[df.fitflag == '1']
    nr_fit = df_fit.shape[0]
    fitted_par = df_fit.parname.values
    ## slice of full df
    df_fit_s = df_fit.iloc[:,0:1]
    ## add index to use in temponest
    df_fit_s['fit_index'] = np.arange(1,nr_fit+1,1)
    df_fit_sort = df_fit_s.sort_values(by='parname')
    return df_fit_sort, nr_fit, fitted_par   


def parnames_compare(df1,df2):
    ## remove the jumps from dataframes, will be analysed separately
    df1 = df1[~df1.parname.str.contains('JUMP')]
    df2 = df2[~df2.parname.str.contains('JUMP')]
    parnames1 = df1.parname.values
    parnames2 = df2.parname.values
    ## parameters in par1, not in par2
    p12 = list(set(parnames1) - set(parnames2))
    ## parameters in par2, not in par1
    p21 = list(set(parnames2) - set(parnames1))
    
    fit1_names = df1[df1.fitflag == '1'].parname.values
    fit2_names = df2[df2.fitflag == '1'].parname.values
    
    ## parameters fitted in par1, not in par2
    f12 = list(set(fit1_names) - set(fit2_names))
    ## parameters fitted in par2, not in par1
    f21 = list(set(fit2_names) - set(fit1_names))
    
    return p12,p21,f12,f21

def jumps_par(df):
    
    dfj = df[df.parname.str.contains('JUMP')]
    nr_jumps = dfj.shape[0]
    nr_fitted_jumps = dfj[dfj.iloc[: , -1] == 1].shape[0]
    return dfj, nr_jumps,nr_fitted_jumps

if __name__ == "__main__":

    filen1 = args.parfiles[0]
    filen2 = args.parfiles[1]
    par_to_tempfiles(filen1,filen2)

    df1, df2 = file_to_frame('temp1.csv', 'temp2.csv')

    os.remove('temp1.csv')
    os.remove('temp2.csv')
   
    check_psrnames(df1,df2)

    p12, p21, f12, f21 = parnames_compare(df1,df2)

    df_fit1, nr1, names1 = fit_par(df1)
    df_fit2, nr2, names2 = fit_par(df2)

    dfj1, nr_jumps1,nr_fitted_jumps1 = jumps_par(df1)
    dfj2, nr_jumps2,nr_fitted_jumps2 = jumps_par(df2)
    
    """Print Summary"""
    
    print("--------- INCLUDED PARAMETERS ---------\n")
    print("Params IN %s and NOT IN %s:" %(filen1, filen2))
    if p12 == []:
        print('None')
    else:
        print(*sorted(p12), sep='\n')
    print("\n")

    print("Params IN %s and NOT IN %s:" %(filen2, filen1))
    if p21 == []:
        print('None')
    else:
        print(*sorted(p21), sep='\n')
    print("\n")


    print("--------- FITTED (FLAG=1) PARAMETERS ---------\n")
    print(f'Number of params fitted in {filen1}:\t {nr1}')
    print(f'Number of params fitted in {filen2}:\t {nr2}')
    print("\n")

    print("Params FITTED IN %s and NOT IN %s:" %(filen1, filen2))
    if f12 == []:
        print('None')
    else:
        print(*sorted(f12), sep='\n')
    print("\n")

    print("Params FITTED IN %s and NOT IN %s:" %(filen2, filen1))
    if f21 == []:
        print('None')
    else:
        print(*sorted(f21), sep='\n')
    print("\n")

    ## verbose mode
    if args.verbose:

        print(f'--- Fit flag params in {filen1} ---')
        print('--- indices for use in temponest priors ---\n')
        print(df_fit1.to_string(header=['Name','Idx'], index=False))
        print("\n")

        print(f'--- Fit flag params in {filen2} ---')
        print('--- indices for use in temponest priors ---\n')
        print(df_fit2.to_string(header=['Name','Idx'], index=False))
        print("\n")

    print(f'Number of JUMPS in {filen1} FITTED/INCLUDED:\t {nr_fitted_jumps1}/{nr_jumps1}')
    print(f'Number of JUMPS in {filen2} FITTED/INCLUDED:\t {nr_fitted_jumps2}/{nr_jumps2}')
    print("\n")

    if args.verbose:

        print(f'--- JUMPS in {filen1} ---\n')
        print(dfj1.to_string(header=False, index=False))
        print("\n")
        print("\n")
        print(f'--- JUMPS in {filen2} ---\n')
        print(dfj2.to_string(header=False, index=False))
