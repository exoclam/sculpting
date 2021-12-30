"""
This is the main function for the transit simulation machinery. It uses general helper functions from simulate_helpers, 
detection helper functions like the Christiansen S/N equation from simulate_detection, and the transit workhorse code
from simulate_transit. After calculating transits, we tally them up and calculate likelihoods in main.

model_van_eylen() --> calculate_transit_me_with_amd() --> logLs
"""

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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from simulate_transit import * 
from simulate_helpers import *
#from simulate_transit import model_van_eylen

### variables for HPG
#task_id = os.getenv('SLURM_ARRAY_TASK_ID')
#path = '/blue/sarahballard/c.lam/sculpting/'

### variables for local
path = '/Users/chrislam/Desktop/sculpting/' # new computer has different username
berger_kepler = pd.read_csv(path+'berger_kepler_stellar17.csv') # crossmatched with Gaia via Bedell
pnum = pd.read_csv(path+'pnum_plus_cands.csv')
pnum = pnum.drop_duplicates(['kepid'])
k = pnum.koi_count.value_counts() 
k = pd.Series([len(berger_kepler)-np.sum(k), 244, 51, 12, 8, 1]) 
G = 6.6743e-8 # gravitational constant in cgs

def better_loglike(lam, k):
    """
    Calculate Poisson log likelihood
    Changed 0 handling from simulate.py to reflect https://www.aanda.org/articles/aa/pdf/2009/16/aa8472-07.pdf

    Params: 
    - lam: model predictions for transit multiplicity (list of ints)
    - k: Kepler transit multiplicity (list of ints); can accept alternate ground truths as well

    Returns: Poisson log likelihood (float)
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


def loglike_direct_draw_better(cube, ndim, nparams, k):
    """
    Run model per hyperparam draw and calculate Poisson log likelihood
    2nd iteration of bridge function between model_direct_draw() and better_logllike()
    Includes geometric transit multiplicity and 0 handling.
    Commented out the zero handling because it's wrong.

    Params: 
    - cube: hyperparam cube of slope and intercept
    - ndim: number of dimensions
    - nparams: number of parameters
    - k: from Berger et al 2020
    Returns: Poisson log-likelihood
    """

    # retrieve prior cube and feed prior-normalized hypercube into model to generate transit multiplicities
    lam, geom_lam, transits, intact_fractions, amds, eccentricities, inclinations_degrees = model_direct_draw(cube)
    #lam = [1e-12 if x==0.0 else x for x in lam] # avoid -infs in logL by turning 0 lams to 1e-12
    #geom_lam = [1e-12 if x==0.0 else x for x in geom_lam] # ditto
    logL = better_loglike(lam, k)
    geom_logL = better_loglike(geom_lam, k)
    
    return logL, lam, geom_lam, geom_logL, transits, intact_fractions, amds, eccentricities, inclinations_degrees


### use fiducial values of m, b, cutoff for now to test eccentricity models
m = -0.3
b = 0.5
cutoff = 1e10 # yrs
frac = 0.4 # fraction of FGK dwarfs with planets
cube = [m, b, cutoff, frac]

berger_kepler_planets = model_van_eylen(berger_kepler.iso_age, berger_kepler, 'limbach', cube)
transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
transit_multiplicity = transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
print(transit_multiplicity)