import pymultinest
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

path = '/Users/chrislam/Desktop/sculpting/' 
berger_kepler = pd.read_csv(path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell, previously berger_kepler_stellar17.csv
pnum = pd.read_csv(path+'pnum_plus_cands_fgk.csv') # previously pnum_plus_cands.csv
pnum = pnum.drop_duplicates(['kepid'])
k = pnum.koi_count.value_counts() 
#k = pd.Series([len(berger_kepler)-np.sum(k), 244, 51, 12, 8, 1]) # 20K stars from 1 Gyr age error cuts
k = pd.Series([len(berger_kepler)-np.sum(k), 833, 134, 38, 15, 5]) # 60K stars from 0.56 fractional age error cuts
k = list(k) # NOTE: THIS INCLUDES THE ZERO BIN
G = 6.6743e-8 # gravitational constant in cgs
#ndim = 4
#nparams = 4

def prior(cube, ndim, nparams):
	"""
	PyMultinest feeds in a unit n-dimensional hypercube. So transform cube[n] by whatever equation s.t. 0-->lower and 1--> upper
	- Cube has the following dimensions:
	0: slope [dex, but sample linearly]
	1: initial probability [sample linearly]
	2: dynamical sculpting turnoff time [yr, sample from logspace]
	3: fraction of systems with intrinsically at least one planet [sample linearly]
	"""
	#cube[0] = -1e-9*np.logspace(8,10,11)[gi_m] # convert from year to Gyr
	cube[0] = cube[0]*2 - 2 # linear from -2 to 0
	cube[1] = cube[1] # linear from 0 to 1
	cube[2] = 10**(cube[2]*2 + 8) # log from 10^8 to 10^10
	#cube[2] = np.logspace(8,10,11)[gi_c] # in Ballard et al in prep, they use log(yrs) instead of drawing yrs from logspace
	cube[3] = cube[3] # linear from 0 to 1

	return cube

def model(iso_age, berger_kepler, model_flag, cube):
	# wrap model_van_eylen() for convenience within the pymultinest framework
	berger_kepler_planets = model_van_eylen(iso_age, berger_kepler, model_flag, cube)
	transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
	transit_multiplicity = transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
	transit_multiplicity = list(transit_multiplicity*cube[3])

	return transit_multiplicity

def loglike(cube, ndim, nparams):
	"""
	Calculate Poisson log likelihood
	The big difference here is that we run the models inside this function. So loglike will be the main driver.

	Params: 
	- lam: model predictions for transit multiplicity (list of ints)
	- k: Kepler transit multiplicity (list of ints); can accept alternate ground truths as well

	Returns: Poisson log likelihood (float)
	"""

	# parameters that I'd like to feed into loglike, but I haven't yet looked into how PyMultinest treats its loglike() function
	model_flag = 'limbach-hybrid'

	# run model and output transit multiplicity to feed into logL machinery as lambda
	lam = model(berger_kepler.iso_age, berger_kepler, model_flag, cube)

	# actually calculate logL
	logL = []
	#print(lam)
	for i in range(len(lam)):
		if lam[i]==0: 	# Changed 0 handling from simulate.py to reflect https://www.aanda.org/articles/aa/pdf/2009/16/aa8472-07.pdf   
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


# number of dimensions our problem has
parameters = ['m','b','c','f']
nparams = len(parameters)


# run MultiNest
pymultinest.run(loglike, prior, nparams, outputfiles_basename=path+'pymultinest/test' + '_1_', resume = False, verbose = True)
json.dump(parameters, open(path+'pymultinest/test' + '_1_params.json', 'w')) # save parameter names


# plot the distribution of a posteriori possible models
plt.figure() 
plt.scatter(np.arange(6), k, '+ ', color='red', label='data')
a = pymultinest.Analyzer(outputfiles_basename=path+'pymultinest/test' + '_1_', n_params = nparams)
for (m, b, c, f) in a.get_equal_weighted_posterior()[::100,:-1]:
	try:
		plt.plot(np.arange(6), model(berger_kepler.iso_age, berger_kepler, model_flag, [m,b,c,f]), '-', color='blue', alpha=0.3, label='model')
	except:
		plt.plot(np.arange(7), model(berger_kepler.iso_age, berger_kepler, model_flag, [m,b,c,f]), '-', color='blue', alpha=0.3, label='model')

plt.savefig(path+ 'pymultinest/test' + '_1_posterior.pdf')
plt.close()


a_lnZ = a.get_stats()['global evidence']
print()
print( '************************')
print( 'MAIN RESULT: Evidence Z ')
print( '************************')
print( '  log Z for model with 1 line = %.1f' % (a_lnZ / np.log10(10)))
print()