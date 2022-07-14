"""
This is the home of the main function for the transit simulation machinery. It uses general helper functions from simulate_helpers and a transit workhorse code
from simulate_transit to output transit statuses for simulated systems. Likelihood_main.py will read in those files and compute logL. 
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
import matplotlib.pyplot as plt
from operator import add
#from simulate_transit import model_van_eylen

### variables for HPG
#task_id = os.getenv('SLURM_ARRAY_TASK_ID')
#path = '/blue/sarahballard/c.lam/sculpting2/'

### variables for local
path = '/Users/chrislam/Desktop/sculpting/' # new computer has different username
berger_kepler = pd.read_csv(path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell, previously berger_kepler_stellar17.csv
#berger_kepler = pd.read_csv(path+'berger_kepler_stellar_k.csv') # K dwarfs only, for comparison with Moriarty & Ballard
pnum = pd.read_csv(path+'pnum_plus_cands_fgk.csv') # previously pnum_plus_cands.csv
pnum = pnum.drop_duplicates(['kepid'])
k = pnum.koi_count.value_counts() 
print(k)
#k = pd.Series([len(berger_kepler)-np.sum(k), 244, 51, 12, 8, 1]) 
#k = pd.Series([len(berger_kepler)-np.sum(k), 833, 134, 38, 15, 5])
#k = [833, 134, 38, 15, 5, 0]
G = 6.6743e-8 # gravitational constant in cgs

def prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c):
    """
    Each model run will use an evenly spaced (m,b, cutoff) tuple on a discrete 11x11x11 3D grid
    We're doing log(time), so slope is sampled linearly (everything gets logged together later)
    If a cutoff results in a zero probability, don't bother 

    gi_m: grid index on m axis
    gi_b: grid index on b axis
    gi_c: grid index for cutoff time axis
    """
    #cube[0] = -1e-9*np.logspace(8,10,11)[gi_m] # convert from year to Gyr
    cube[0] = np.linspace(-2,0,11)[gi_m] 
    cube[1] = np.linspace(0,1,11)[gi_b]
    #cube[2] = np.logspace(1e8,1e10,11)
    cube[2] = np.logspace(8,10,11)[gi_c] # in Ballard et al in prep, they use log(yrs) instead of drawing yrs from logspace
    return cube

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

def sanity_check(model_flag):
    """
    The most important unit test so far...making sure that model_van_eylen and model_vectorized are producing the same transit multiplicities. 
    For some reason, model_van_eylen is consistent, while model_vectorized keeps trading off its 6-bin into the 3- and 4-bins.

    """

    ### use fiducial values of m, b, cutoff, and frac for now to test eccentricity models
    m = -0.2 #-1.2
    b = 0.9
    cutoff = 1e9 # yrs
    frac = 0.2 # fraction of FGK dwarfs with planets
    cube = [m, b, cutoff, frac]
    print("cube: ", cube)

    ### VECTORIZED
    berger_kepler_planets = model_vectorized(berger_kepler, model_flag, cube)
    """
    THIS IS WHERE I FIGURED OUT THAT RANDOM.CHOICE WAS TOTALLY ASSIGNING 5 OR 6 FOR ALL INTACT SYSTEMS, AND 1 OR 2 FOR DISRUPTED
    print("1s: ", len(berger_kepler_planets.loc[berger_kepler_planets.num_planets==1]))
    print("2s: ", len(berger_kepler_planets.loc[berger_kepler_planets.num_planets==2]))
    print("5s: ", len(berger_kepler_planets.loc[berger_kepler_planets.num_planets==5]))
    print("6s: ", len(berger_kepler_planets.loc[berger_kepler_planets.num_planets==6]))
    print("intact: ", len(berger_kepler_planets.loc[berger_kepler_planets.intact_flag=='intact']))
    print("disrupted: ", len(berger_kepler_planets.loc[berger_kepler_planets.intact_flag=='disrupted']))
    #plt.hist(berger_kepler_planets.num_planets)
    #plt.show()
    quit()
    """
    transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
    transit_multiplicity = list(frac*transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
    transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
    #berger_kepler_planets.to_csv('transits02_04_04_25.csv')
    
    # make sure the 6-multiplicity bin is filled in with zero and ignore zero-bin
    #k[6] = 0
    #k = k[1:].reset_index()[0]
    print("transit multiplicity vectorized: ", transit_multiplicity)

    ### VAN EYLEN
    berger_kepler_planets = model_van_eylen(berger_kepler.iso_age, berger_kepler, model_flag, cube)
    transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
    transit_multiplicity = list(frac*transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
    transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
    #berger_kepler_planets.to_csv('transits02_04_04_25.csv')
    
    # make sure the 6-multiplicity bin is filled in with zero and ignore zero-bin
    #k[6] = 0
    #k = k[1:].reset_index()[0]
    print("transit multiplicity van eylen: ", transit_multiplicity)

    return

def unit_test(k, model_flag):

    ### use fiducial values of m, b, cutoff, and frac for now to test eccentricity models
    m = -0.2
    b = 0.3 # 0.9
    cutoff = 6.309573e+08 # 1e9 # yrs
    frac = 0.9 # fraction of FGK dwarfs with planets

    m = -0.4 # -3.65967387e-01  
    b = 8.88934417e-01  
    cutoff = 5e9 # 6.31628913e+09  
    frac = 0.35
    cube = [m, b, cutoff]
    print("cube: ", cube)

    for j in range(30):

        #berger_kepler_planets = model_van_eylen(berger_kepler.iso_age, berger_kepler, model_flag, cube)
        berger_kepler_planets = model_vectorized(berger_kepler, model_flag, cube)
        #plt.hist(berger_kepler_planets.sn)
        #plt.show()

        # plot detected ratio as a function of SNR, and of period
        bins = np.linspace(2, 300, 20) # period
        #bins = np.linspace(0, 100, 50) # S/N
        print("periods: ", bins)
        berger_kepler_planets['digitized_P'] = np.digitize(berger_kepler_planets.P, bins=bins)
        #berger_kepler_planets['digitized_sn'] = np.digitize(berger_kepler_planets.sn, bins=bins)
        #print(np.sort(berger_kepler_planets['digitized_P'].unique()))
        #bin_means = [np.nanmean(berger_kepler_planets.loc[berger_kepler_planets.digitized_P == i].sn) for i in np.sort(berger_kepler_planets.digitized_P.unique())]
        #print("mean SNs: ", bin_means)
    
        detected = []
        for i in np.sort(berger_kepler_planets.digitized_P.unique()):
            sub = berger_kepler_planets.loc[berger_kepler_planets.digitized_P == i]

            transiters_berger_kepler = sub.loc[sub['transit_status']==1] # use this for 1+ bins only
            geom_transiters_berger_kepler = sub.loc[sub['geom_transit_status']==1] # use this for 1+ bins only

            transit_multiplicity = list(frac*transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
            geom_transit_multiplicity = list(frac*geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)
        
            transit_multiplicity = np.array(transit_multiplicity)
            geom_transit_multiplicity = np.array(geom_transit_multiplicity)
            #print(i, transit_multiplicity, geom_transit_multiplicity)
            detected.append(np.sum(transit_multiplicity)/np.sum(geom_transit_multiplicity))


        print(bins, detected)
        print(len(bins), len(detected))
        plt.scatter(bins[:-1], detected)
    plt.ylabel('detection rate')
    plt.xlabel('period [days]')
    plt.show()
    quit()

    transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1] # use this for 1+ bins only
    geom_transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['geom_transit_status']==1] # use this for 1+ bins only

    ## use this when applying frac post-hoc
    if len(cube)==3:
        transit_multiplicity = list(frac*transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
        geom_transit_multiplicity = list(frac*geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)
    elif len(cube)==4:
        transit_multiplicity = list(berger_kepler_planets.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
        geom_transit_multiplicity = list(berger_kepler_planets.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)

    transit_multiplicity = np.array(transit_multiplicity)
    geom_transit_multiplicity = np.array(geom_transit_multiplicity)
    print("detected ratios: ", transit_multiplicity/geom_transit_multiplicity)

    #print([0.] * (len(k) - len(transit_multiplicity)))
    #transit_multiplicity += [0.] * (len(k) - len(transit_multiplicity)) # pad with zeros to match length of k
    #geom_transit_multiplicity += [0.] * (len(k) - len(geom_transit_multiplicity)) # pad with zeros to match length of k
    transit_multiplicity = pad(transit_multiplicity)
    geom_transit_multiplicity = pad(geom_transit_multiplicity)
    #berger_kepler_planets.to_csv('transits02_04_04_25.csv')

    # pad zero bin in front
    #print(len(berger_kepler_planets.kepid.unique())-np.sum(transit_multiplicity))
    #transit_multiplicity.insert(0, len(berger_kepler_planets.kepid.unique())-np.sum(transit_multiplicity))
    #print(transit_multiplicity)
    #quit()

    # make sure the 6-multiplicity bin is filled in with zero and ignore zero-bin
    #k[6] = 0
    #k = k[1:].reset_index()[0]
    print("transit multiplicity: ", transit_multiplicity)
    print("geometric transit multiplicity: ", geom_transit_multiplicity)
    print("k: ", list(k))

    # get intact and disrupted fractions (combine them later to get fraction of systems w/o planets)
    intact = berger_kepler_planets.loc[berger_kepler_planets.intact_flag=='intact']
    disrupted = berger_kepler_planets.loc[berger_kepler_planets.intact_flag=='disrupted']
    intact_frac_of_hosts = len(intact.kepid.unique())/len(berger_kepler_planets.kepid.unique())
    disrupted_frac_of_hosts = len(disrupted.kepid.unique())/len(berger_kepler_planets.kepid.unique())
    intact_frac_overall = intact_frac_of_hosts*frac
    disrupted_frac_overall = disrupted_frac_of_hosts*frac 
    print("intact frac: ", intact_frac_of_hosts)
    print("disrupted frac: ", disrupted_frac_of_hosts)
    print("intact frac overall: ", intact_frac_overall)
    print("disrupted frac overall: ", disrupted_frac_overall)


    # get transit and geometric transit multiplicities for intact vs disrupted populations
    transiters_intact = intact.loc[intact['transit_status']==1] # use this for 1+ bins only
    transiters_disrupted = disrupted.loc[disrupted['transit_status']==1] # use this for 1+ bins only
    geom_transiters_intact = intact.loc[intact['geom_transit_status']==1] # use this for 1+ bins only
    geom_transiters_disrupted = disrupted.loc[disrupted['geom_transit_status']==1] # use this for 1+ bins only

    if len(cube)==3:
        transit_multiplicity_intact = list(frac*transiters_intact.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
        transit_multiplicity_disrupted = list(frac*transiters_disrupted.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
        geom_transit_multiplicity_intact = list(frac*geom_transiters_intact.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)
        geom_transit_multiplicity_disrupted = list(frac*geom_transiters_disrupted.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)
    elif len(cube)==4:
        transit_multiplicity_intact = list(intact.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
        transit_multiplicity_disrupted = list(disrupted.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
        geom_transit_multiplicity_intact = list(intact.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)
        geom_transit_multiplicity_disrupted = list(disrupted.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)

    #transit_multiplicity_intact += [0.] * (len(k) - len(transit_multiplicity_intact)) # pad with zeros to match length of k
    #transit_multiplicity_disrupted += [0.] * (len(k) - len(transit_multiplicity_disrupted)) # pad with zeros to match length of k
    #geom_transit_multiplicity_intact += [0.] * (len(k) - len(geom_transit_multiplicity_intact)) # pad with zeros to match length of k
    #geom_transit_multiplicity_disrupted += [0.] * (len(k) - len(geom_transit_multiplicity_disrupted)) # pad with zeros to match length of k
    transit_multiplicity_intact = pad(transit_multiplicity_intact)
    transit_multiplicity_disrupted = pad(transit_multiplicity_disrupted)
    geom_transit_multiplicity_intact = pad(geom_transit_multiplicity_intact)
    geom_transit_multiplicity_disrupted = pad(geom_transit_multiplicity_disrupted)

    print("")
    print("intact transit multiplicity: ", transit_multiplicity_intact)
    print("disrupted transit multiplicity: ", transit_multiplicity_disrupted)
    print("intact geometric transit multiplicity: ", geom_transit_multiplicity_intact)
    print("disrupted geometric transit multiplicity: ", geom_transit_multiplicity_disrupted)

    return transit_multiplicity, geom_transit_multiplicity, transit_multiplicity_intact, transit_multiplicity_disrupted, geom_transit_multiplicity_intact, geom_transit_multiplicity_disrupted

    # calculate log likelihood
    logL = better_loglike(list(transit_multiplicity), list(k))
    print("logL: ", logL)
    print("old logL: ", better_loglike([466.8, 72.8, 14.8, 13.2, 7.6, 2.4], k))
    print("total: ", len(berger_kepler_planets))
    print("transiters: ", len(transiters_berger_kepler))

    # redundancy check
    redundant = redundancy_check(m, b, cutoff)
    print("redundancy check: ", redundant)

    """
    # sanity check the resulting {e, i} distribution with a plot
    ecc = berger_kepler_planets.ecc
    incl = np.abs(berger_kepler_planets.incl*180/np.pi)
    plt.scatter(ecc, incl, s=2)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e-3,1e0)
    plt.ylim(1e-2,1e2)
    plt.savefig('ecc-inc-test.png')
    """
    return berger_kepler_planets.ecc, np.abs(berger_kepler_planets.mutual_incl)*180/np.pi, berger_kepler_planets

"""
#### Compare transit multiplicity outputs of model_van_eylen vs model_vectorized
sanity_check('limbach-hybrid')
quit()
"""

#### Test vectorized approach for time
start = datetime.now()
iterations = 5
transit_multiplicities = []
geom_transit_multiplicities = []
transit_multiplicities_intact = []
geom_transit_multiplicities_intact = []
transit_multiplicities_disrupted = []
geom_transit_multiplicities_disrupted = []

for i in range(iterations):
    transit_multiplicity, geom_transit_multiplicity, transit_multiplicity_intact, transit_multiplicity_disrupted, geom_transit_multiplicity_intact, geom_transit_multiplicity_disrupted = unit_test(k, 'limbach-hybrid')
    transit_multiplicities.append(transit_multiplicity)
    geom_transit_multiplicities.append(geom_transit_multiplicity)
    transit_multiplicities_intact.append(transit_multiplicity_intact)
    geom_transit_multiplicities_intact.append(geom_transit_multiplicity_intact)
    transit_multiplicities_disrupted.append(transit_multiplicity_disrupted)
    geom_transit_multiplicities_disrupted.append(geom_transit_multiplicity_disrupted)

print(np.mean(transit_multiplicities, axis=0))
print(np.mean(geom_transit_multiplicities, axis=0))

plt.fill_between(np.arange(7)[1:], np.min(transit_multiplicities, axis=0), np.max(transit_multiplicities, axis=0), alpha=0.5, color='orange')
plt.scatter(np.arange(7)[1:], np.mean(transit_multiplicities, axis=0), c='orange', label='detected')
plt.fill_between(np.arange(7)[1:], np.min(geom_transit_multiplicities, axis=0), np.max(geom_transit_multiplicities, axis=0), alpha=0.5, color='k')
plt.scatter(np.arange(7)[1:], np.mean(geom_transit_multiplicities, axis=0), c='k', label='geometric')
plt.xlabel('transit multiplicity')
plt.ylabel('number of stars')
plt.legend()
plt.show()

plt.fill_between(np.arange(7)[1:], np.min(transit_multiplicities_intact, axis=0), np.max(transit_multiplicities_intact, axis=0), alpha=0.5, color='orange')
plt.scatter(np.arange(7)[1:], np.mean(transit_multiplicities_intact, axis=0), c='orange', label='detected')
plt.fill_between(np.arange(7)[1:], np.min(geom_transit_multiplicities_intact, axis=0), np.max(geom_transit_multiplicities_intact, axis=0), alpha=0.5, color='k')
plt.scatter(np.arange(7)[1:], np.mean(geom_transit_multiplicities_intact, axis=0), c='k', label='geometric')
plt.xlabel('transit multiplicity')
plt.ylabel('number of stars')
plt.legend()
plt.show()

plt.fill_between(np.arange(7)[1:], np.min(transit_multiplicities_disrupted, axis=0), np.max(transit_multiplicities_disrupted, axis=0), alpha=0.5, color='orange')
plt.scatter(np.arange(7)[1:], np.mean(transit_multiplicities_disrupted, axis=0), c='orange', label='detected')
plt.fill_between(np.arange(7)[1:], np.min(geom_transit_multiplicities_disrupted, axis=0), np.max(geom_transit_multiplicities_disrupted, axis=0), alpha=0.5, color='k')
plt.scatter(np.arange(7)[1:], np.mean(geom_transit_multiplicities_disrupted, axis=0), c='k', label='geometric')
plt.xlabel('transit multiplicity')
plt.ylabel('number of stars')
plt.legend()
plt.show()

#"""

#### Compare CDPP sampling methods

#### Run unit test to plot and compare different eccentricity distribution assumptions
"""
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
models = ['rayleigh', 'beta', 'half-Gaussian', 'mixed', 'limbach', 'limbach-hybrid']
plot_i = 0
for row in range(2):
    for column in range(3):
        model_flag = models[plot_i]
        ecc, incl, __ = unit_test(k, model_flag)
        quit()
        ax = plt.subplot2grid((2,3), (row,column))
        im = ax.hexbin(ecc, incl, yscale='log', xscale='log', extent=(-3, 0, -2, 2))
        fig2 = sns.kdeplot(np.array(ecc), np.array(incl), legend = True, levels=[0.68, 0.95], colors=['black','red'])
        #plt.hexbin(berger_kepler_planets.ecc, np.log10(berger_kepler_planets.incl*180/np.pi))  
        ax.set_ylim(1e-2, 1e2)
        ax.set_xlim(1e-3, 1e0)
        ax.set_title(model_flag)
        if plot_i==3:
            ax.set_xlabel('eccentricity')
        if plot_i==0:
            ax.set_ylabel('inclination')
        fig.colorbar(im, ax=ax)
        #plt.savefig('ecc-inc-limbach-00-005.png')

        plot_i += 1

#plt.savefig('ecc-inc-00-05.png')
"""

end = datetime.now()
print("ELAPSED: ", end-start)
quit()

#### Run unit test to plot AMD over {ecc, inc}

__, __, df = unit_test(k, 'limbach-hybrid')
df['amd_products'] = df.lambda_ks*df.second_terms
#print(df.ecc)
#print(df.mutual_incl)
#print(df.amd_products)
#df = df.loc[df.mutual_incl*180/np.pi >= 1] # TEMPORARY FOR TESTING

# needed for the groupby.mean() to work
df.ecc = df.ecc.astype(float) 
df.mutual_incl = df.mutual_incl.astype(float)

df_amd = df.groupby('kepid').sum('amd_products')

#df_ecc = df.groupby('kepid').mean('ecc')
df_amd['mutual_incl'] = df.groupby(['kepid'])['mutual_incl'].mean()

#df_incl = df.groupby('kepid').mean('mutual_incl')
df_amd['ecc'] = df.groupby(['kepid'])['ecc'].mean()

#df_amd['ecc'] = df_ecc['ecc']
#df_amd['mutual_incl'] = df_incl['mutual_incl']
#df_amd = df_amd.loc[df_amd.amd_products > 2e48] # TEMPORARY FOR TESTING
#df_ecc = df_ecc.loc[df_ecc.amd_products > 1e48] # TEMPORARY FOR TESTING
#df_incl = df_incl.loc[df_incl.amd_products > 1e48] # TEMPORARY FOR TESTING

"""
print(len(df_amd), len(df_ecc), len(df_incl))
print(df_amd)
print(df_ecc)
#print(len(df_amd))
#print(min(df_amd.amd_products), max(df_amd.amd_products))
print(df_amd.ecc, df_ecc.ecc, df_incl.ecc)
"""

#plt.hist(df_amd.amd_products)
#plt.show()

fig = plt.figure(figsize=(10, 6))
#plt.set_cmap("Blues_r")
plt.scatter(df_amd.ecc, df_amd.mutual_incl*180/np.pi, s=2, c=np.log10(df_amd.amd_products))
plt.yscale('log')
plt.xscale('log')
plt.xlim(1e-3,1e0)
plt.ylim(1e-2,2e2)
#plt.xlabel('eccentricity', fontsize=28)
plt.ylabel('mutual inclination [deg]', fontsize=28)
plt.tick_params(axis='both', labelsize=26)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=22)
cbar.set_label(label='log AMD', size=24, labelpad=20)
fig.tight_layout()
#plt.show()
plt.savefig('/Users/chrislam/Desktop/sculpting/poster_plots/amd_plot1.pdf', bbox_inches='tight', format='pdf')
