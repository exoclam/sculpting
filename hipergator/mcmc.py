import emcee
import corner
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
import sys
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from simulate_transit import * 
from simulate_helpers import *
import datetime
from schwimmbad import MPIPool

path = '/blue/sarahballard/c.lam/sculpting2/' # HPG
#path = '/Users/chrislam/Desktop/sculpting/' 
berger_kepler = pd.read_csv(path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell, previously berger_kepler_stellar17.csv
pnum = pd.read_csv(path+'pnum_plus_cands_fgk.csv') # previously pnum_plus_cands.csv
pnum = pnum.drop_duplicates(['kepid'])
#k = pnum.koi_count.value_counts() 
#k = pd.Series([len(berger_kepler)-np.sum(k), 244, 51, 12, 8, 1]) # 20K stars from 1 Gyr age error cuts
#k = pd.Series([len(berger_kepler)-np.sum(k), 833, 134, 38, 15, 5]) # 60K stars from 0.56 fractional age error cuts
k = [833, 134, 38, 15, 5, 0]
#k = list(k) # NOTE: THIS INCLUDES THE ZERO BIN
G = 6.6743e-8 # gravitational constant in cgs

ndim = 3

def prior(cube):
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
	a, b = 0.01, 1.25
	#cube[2] = scipy.stats.loguniform.ppf(cube[2], a, b) # log the cutoff parameter
	cube[2] = 10**(cube[2]*2 + 8) # log? from 10^8 to 10^10
	#cube[2] = np.logspace(8,10,11)[gi_c] # in Ballard et al in prep, they use log(yrs) instead of drawing yrs from logspace
	#cube[3] = cube[3]*0.4 # linear from 0 to 0.4
	return cube

def prior_array(cube):
	cube[:, 0] = cube[:, 0]*2 - 2 # linear from -2 to 0
	cube[:, 1] = cube[:, 1] # linear from 0 to 1
	cube[:, 2] = 10**(cube[:, 2]*2 + 8)  # log from 10^8 to 10^10

	return cube

def log_prior(cube):
	m, b, log_c = cube

	if -2 <= m <= 0 and 0 <= b <= 1 and 8 <= log_c <= 10:
		return 0

	return -np.inf 

def make_model(iso_age, berger_kepler, model_flag, cube):
	# wrap model_van_eylen() for convenience within the pymultinest framework
	#berger_kepler_planets = model_van_eylen(iso_age, berger_kepler, model_flag, cube)
	f = 0.2
	berger_kepler_planets = model_vectorized(berger_kepler, model_flag, cube)
	transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
	transit_multiplicity = list(f*transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
	transit_multiplicity += [0.] * (len(k) - len(transit_multiplicity)) # pad with zeros to match length of k

	return transit_multiplicity

def loglike(cube, k):
	"""
	Calculate Poisson log likelihood
	The big difference here is that we run the models inside this function. So loglike will be the main driver.

	Params: 
	- lam: model predictions for transit multiplicity (list of ints)
	- k: Kepler transit multiplicity (list of ints); can accept alternate ground truths as well

	Returns: Poisson log likelihood (float)
	"""
	#print("CUBE: ", cube, cube[0])
	np.savetxt(file, np.array((cube)), fmt='%f', delimiter='\t', newline='\t')

	# parameters that I'd like to feed into loglike, but I haven't yet looked into how PyMultinest treats its loglike() function
	model_flag = 'limbach-hybrid'

	try:
		# run model and output transit multiplicity to feed into logL machinery as lambda
		lam = make_model(berger_kepler.iso_age, berger_kepler, model_flag, cube)
		np.savetxt(file, np.array([lam]), fmt='%f', delimiter=',', newline='\t')

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

		#print(np.sum(logL))
		np.savetxt(file, np.array([np.sum(logL)]), fmt='%f', newline='\n')

	except:
		return -1e5

	return np.sum(logL)

def log_probability(cube):
	"""
	Calculate Poisson log likelihood
	The big difference here is that we run the models inside this function. So loglike will be the main driver.

	Params: 
	- lam: model predictions for transit multiplicity (list of ints)
	- k: Kepler transit multiplicity (list of ints); can accept alternate ground truths as well

	Returns: Poisson log likelihood (float)
	"""
	#print("CUBE: ", cube, cube[0])
	###np.savetxt(file, np.array((cube)), fmt='%f', delimiter='\t', newline='\t')

	lp = log_prior(cube)
	if not np.isfinite(lp):
		return -np.inf

	return lp + log_likelihood(cube)

def log_likelihood(cube):
	cube = [cube[0],cube[1],10**cube[2]] # unlog cutoff time

	# parameters that I'd like to feed into loglike, but I haven't yet looked into how PyMultinest treats its loglike() function
	model_flag = 'limbach-hybrid'

	# run model and output transit multiplicity to feed into logL machinery as lambda
	lam = make_model(berger_kepler.iso_age, berger_kepler, model_flag, cube)
	###np.savetxt(file, np.array([lam]), fmt='%f', delimiter=',', newline='\t')

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

	#print(np.sum(logL))
	###np.savetxt(file, np.array([np.sum(logL)]), fmt='%f', newline='\n')

	return np.sum(logL)

#start = datetime.datetime.now()

#file = open(path+"hipergator/emcee.txt", "w")
#file.write("m,b,c,transit_multiplicity,logL\n") # header

ndim, nwalkers = 3, 8
ivar = [-0.2, 0.4, 9.6] # initial args from discretized sampling; log_c of 9.6 corresponds to c of 3.98e9 yrs

"""
First go-around
#p0 = np.random.rand(nwalkers, ndim) # initialize args
#print("before: ", p0)
#p0 = prior_array(p0) # make prior
#print("after: ", p0)
"""

pos = ivar + 1e-2 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[k])
state = sampler.run_mcmc(pos, 1000, progress=True) # 100 steps for burn-in

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
#print(samples)

# plot to show when burn-in ends for discarding
labels = ["m", "b", "log(c)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig(path+'short_burn_in.pdf', format='pdf')

tau = sampler.get_autocorr_time()
print("tau: ", tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print("number of samples: ", flat_samples.shape)

np.savetxt(path+'short_samples.txt', flat_samples, fmt='%f', delimiter='\t', newline='\t')

fig = corner.corner(flat_samples, labels=labels)

plt.savefig(path+'short_corner.pdf', format='pdf')

"""
sampler.reset()

sampler.run_mcmc(state, 1000)

samples = sampler.get_chain(flat=True)
end = datetime.datetime.now()

print("time elapsed: ", end-start)

plt.hist(samples[:, 0], 100, color="k", histtype="step")
plt.savefig(path+'mcmc_short.pdf', format='pdf')
"""










