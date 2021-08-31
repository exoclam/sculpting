import json
import sys
import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma
from glob import glob
import os
#import seaborn as sns
import csv
from ast import literal_eval
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

def better_loglike(lam, k):
    """
    Calculate Poisson log likelihood
    Changed 0 handling from simulate.py to reflect https://www.aanda.org/articles/aa/pdf/2009/16/aa8472-07.pdf
    """

    logL = []
    #print(lam)
    for i in range(len(lam)):
        if lam[i]==0:    
            term3 = -lgamma(k[i]+1)
            term2 = -lam[i]
            term1 = 0
            logL.append(term1+term2+term3)
        else:
            term3 = -lgamma(k[i]+1)
            term2 = -lam[i]
            term1 = k[i]*np.log(lam[i])
            logL.append(term1+term2+term3)

    return np.sum(logL)

path = '/Users/chris/Desktop/sculpting/'

berger_kepler = pd.read_csv(path+'berger_kepler_stellar17.csv') # crossmatched with Gaia via Bedell
print(len(berger_kepler))
print(berger_kepler.head())

hist, bins = np.histogram(berger_kepler.iso_age, bins=100)
print(np.std(berger_kepler.iso_age))

# transit multiplicity from Kepler/Gaia Berger et al 2020, plus Bedell, plus Exoplanet Archive
# see isolate_with_bedell.ipynb
pnum = pd.read_csv(path+'pnum_plus_cands.csv')
print(len(pnum))
pnum = pnum.drop_duplicates(['kepid'])
print(len(pnum))
k = pnum.koi_count.value_counts() 
k = pd.Series([len(berger_kepler)-np.sum(k), 244, 51, 12, 8, 1]) 
print(k)

# redo ms because I forgot that I rounded to single decimal beforehand
ms_corrected = []
ms_corrected_for_plotting = []
for gi_m in range(11):
    for gi_b in range(11):
        ms = -1e-9*np.logspace(8,10,11)[gi_m]
        ms_corrected.append(ms)
        ms_for_plotting = np.log10(np.logspace(8,10,11))[gi_m]
        ms_corrected_for_plotting.append(ms_for_plotting)
print(ms_corrected)
print(ms_corrected_for_plotting)

with open('simulations_w_cutoff.csv','r') as csvfile: # simulations_w_logslope.csv
    read_csv = csv.reader(csvfile, delimiter='\t')
    read_csv=list(read_csv)

df = pd.DataFrame(read_csv[1:], columns=read_csv[0])
df.lams = df.lams.apply(literal_eval) # convert back from string to list of floats
print(df.columns, df)
df.logLs = df.logLs.apply(literal_eval) # convert back from string to list of floats
df.bs = df.bs.apply(literal_eval)
#df.ms = df.ms.apply(literal_eval)
#df.ms = ms_corrected
df.intact_fracs = df.intact_fracs.apply(literal_eval)

### re-introduce nonzero-bin transit multiplicities
df_lams_nonzero1 = []
df_lams_nonzero2 = []
df_lams_nonzero3 = []
df_lams_nonzero4 = []
df_lams_nonzero5 = []
df_lams_nonzero6 = []
df_lams_nonzero7 = []
df_lams_nonzero8 = []
df_lams_nonzero9 = []
df_lams_nonzero10 = []
df_lams_nonzero11 = []
df_lams_nonzero = [df_lams_nonzero1, df_lams_nonzero2, df_lams_nonzero3, df_lams_nonzero4, df_lams_nonzero5,
df_lams_nonzero6, df_lams_nonzero7, df_lams_nonzero8, df_lams_nonzero9, df_lams_nonzero10, df_lams_nonzero11]
for x in df.lams:
	for i in range(11):
		df_lams_nonzero[i].append([[y_elt*0.1*float(i) for y_elt in y[1:]] for y in x]) # scale transit multiplicity by varying population-level planet occurrence scaling rate
print(df_lams_nonzero1)
#print(df_lams_nonzero11)
print(len(df_lams_nonzero11))

start = datetime.now()
logLs_nonzero1 = []
logLs_nonzero2 = []
logLs_nonzero3 = []
logLs_nonzero4 = []
logLs_nonzero5 = []
logLs_nonzero6 = []
logLs_nonzero7 = []
logLs_nonzero8 = []
logLs_nonzero9 = []
logLs_nonzero10 = []
logLs_nonzero11 = []
logLs_nonzero = [logLs_nonzero1, logLs_nonzero2, logLs_nonzero3, logLs_nonzero4, logLs_nonzero5,
logLs_nonzero6, logLs_nonzero7, logLs_nonzero8, logLs_nonzero9, logLs_nonzero10, logLs_nonzero11]
for i, df_lams_nonzero_elt in enumerate(df_lams_nonzero):
	temp_logLs_nonzero1 = []
	for x in df_lams_nonzero_elt:
	    temp_logLs_nonzero2 = []
	    for y in x:
	        temp_logLs_nonzero2.append(better_loglike(y, k[1:].reset_index()[0]))
	    temp_logLs_nonzero1.append(temp_logLs_nonzero2)
	logLs_nonzero[i].append(temp_logLs_nonzero1)

print(logLs_nonzero1)
#print(logLs_nonzero11)
print(len(logLs_nonzero11))

end = datetime.now()
print(end-start)

df['lams_nonzero1'] = df_lams_nonzero1
df['lams_nonzero2'] = df_lams_nonzero2
df['lams_nonzero3'] = df_lams_nonzero3
df['lams_nonzero4'] = df_lams_nonzero4
df['lams_nonzero5'] = df_lams_nonzero5
df['lams_nonzero6'] = df_lams_nonzero6
df['lams_nonzero7'] = df_lams_nonzero7
df['lams_nonzero8'] = df_lams_nonzero8
df['lams_nonzero9'] = df_lams_nonzero9
df['lams_nonzero10'] = df_lams_nonzero10
df['lams_nonzero11'] = df_lams_nonzero11

df['logLs_nonzero1'] = logLs_nonzero1[0]
df['logLs_nonzero2'] = logLs_nonzero2[0]
df['logLs_nonzero3'] = logLs_nonzero3[0]
df['logLs_nonzero4'] = logLs_nonzero4[0]
df['logLs_nonzero5'] = logLs_nonzero5[0]
df['logLs_nonzero6'] = logLs_nonzero6[0]
df['logLs_nonzero7'] = logLs_nonzero7[0]
df['logLs_nonzero8'] = logLs_nonzero8[0]
df['logLs_nonzero9'] = logLs_nonzero9[0]
df['logLs_nonzero10'] = logLs_nonzero10[0]
df['logLs_nonzero11'] = logLs_nonzero11[0]

print(df)
df.to_csv('simulations_w_cutoff_and_scaling.csv')