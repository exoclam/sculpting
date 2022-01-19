# for each {m, b, cutoff} filename, read all 9 files
# calculate logLs and get mean and std (or min and max)
# read out to plot in Jupyter locally

import json
import sys
import os
from glob import glob
import numpy as np
import pandas as pd
from math import lgamma
#from simulate_main import prior_grid_logslope, better_loglike
from datetime import datetime

path = '/blue/sarahballard/c.lam/sculpting2/' # HPG
#path = '/Users/chrislam/Desktop/sculpting/' # local

# get ground truth to calculate logLs
berger_kepler = pd.read_csv(path+'berger_kepler_stellar17.csv') # crossmatched with Gaia via Bedell
pnum = pd.read_csv(path+'pnum_plus_cands.csv')
pnum = pnum.drop_duplicates(['kepid'])
k = pnum.koi_count.value_counts() 
k = pd.Series([len(berger_kepler)-np.sum(k), 244, 51, 12, 8, 1, 0]) 
print("k: ", k)

# set up hypercube just so I can associate logLs with correct hyperparams
ndim = 3
nparams = 3
cube = [0, 0, 0]

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

data_path = '/blue/sarahballard/c.lam/sculpting2/simulations2/limbach-hybrid/'
#print("path: ", path)

# group file names by {m, b, cutoff} simulation
# note that some will be empty because I skip over them due to redundancy check from simulate_main.py
sims = []
ms = []
bs = []
cs = []
fs = []
max_logLs = []
min_logLs = []
mean_logLs = []
median_logLs = []
std_logLs = []
transit_multiplicities_all = []
start = datetime.now()
#print("start: ", start)
for gi_m in range(11):
	for gi_b in range(11):
		for gi_c in range(11):

			sim = glob(data_path+'transits'+str(gi_m)+'_'+str(gi_b)+'_'+str(gi_c)+'_'+'*')
			cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)

			# cycle through different fractions of systems with planets
			for f in np.linspace(0, 1, 11):
				ms.append(cube[0])
				bs.append(cube[1])
				cs.append(cube[2])
				fs.append(f)

				logLs = []
				transit_multiplicities = []
				for i in range(len(sim)):
					df = pd.read_csv(sim[i], delimiter=',', names=list(range(150))) # handle the few rows of different lengths; most are 150
					new_header = df.iloc[0] #grab the first row for the header
					df = df[1:] #take the data less the header row
					df.columns = new_header #set the header row as the df header

					# isolate transiting planets
					transiters_berger_kepler = df.loc[df['transit_status']==1]

					# compute transit multiplicity and save off the original transit multiplicity (pre-frac)
					transit_multiplicity = transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
					transit_multiplicities.append(list(transit_multiplicity))

					# calculate logLs for different fracs and keep the best one
					logL = better_loglike(transit_multiplicity*f, k)
					logLs.append(logL)

				transit_multiplicities_all.append(transit_multiplicities)

				try:
					max_logLs.append(max(logLs))
					min_logLs.append(min(logLs))
					mean_logLs.append(np.mean(logLs))
					std_logLs.append(np.std(logLs))
					median_logLs.append(np.median(logLs))

				except: # sometimes logLs will be empty where a redundancy check was passed for some hyperparam tuple
					max_logLs.append([])
					min_logLs.append([])
					mean_logLs.append([])
					std_logLs.append([])
					median_logLs.append([])

				end = datetime.now()
				#print("end: ", end)
				#print("time: ", end-start)
				#quit()

print(len(ms))
print(len(bs))
print(len(cs))
print(len(fs))
print(len(max_logLs))
print(len(min_logLs))
print(len(mean_logLs))
print(len(median_logLs))
print(len(std_logLs))
print(len(transit_multiplicities_all))

df_logL = pd.DataFrame({'ms': ms, 'bs': bs, 'cs': cs, 'fs': fs, 'max_logLs': max_logLs, 'min_logLs': min_logLs, 
	'mean_logLs': mean_logLs, 'median_logLs': median_logLs, 'std_logLs': std_logLs, 
	'transit_multiplicities_all': transit_multiplicities_all})
print(df_logL)
df_logL.to_csv(path+'logLs_fgk.csv', index=False)
