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
#berger_kepler = pd.read_csv(path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
#pnum = pd.read_csv(path+'pnum_plus_cands_fgk.csv')
#pnum = pnum.drop_duplicates(['kepid'])
#k = pnum.koi_count.value_counts() 
#k = pd.Series([len(berger_kepler)-np.sum(k), 244, 51, 12, 8, 1, 0]) 
k = pd.Series([833, 134, 38, 15, 5, 0])
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

def main(filename): # one read and write per sim[i] filename
	try:
		df = pd.read_csv(filename, delimiter=',')
		# isolate transiting planets
		transiters_berger_kepler = df.loc[df['transit_status']==1]

		# compute transit multiplicity and save off the original transit multiplicity (pre-frac)
		transit_multiplicity = f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid

		# calculate logLs for different fracs and keep the best one
		logL = better_loglike(transit_multiplicity, k)

		# get intact and disrupted fractions (combine them later to get fraction of systems w/o planets)
		intact = df.loc[df.intact_flag=='intact']
		disrupted = df.loc[df.intact_flag=='disrupted']
		intact_frac = f*len(intact)/len(df)
		disrupted_frac = f*len(disrupted)/len(df)

		# output
	    #out = np.array((str(m), str(b), str(c), str(f), str(logL), str(list(transit_multiplicity)), str(intact_frac), str(disrupted_frac)))
		out = np.array((filename, m, b, c, f, logL, list(transit_multiplicity), intact_frac, disrupted_frac))
	    #print(out)
		np.savetxt(file1, out, fmt='%s', newline='\t')
		file1.write("\n")

	except:
		print("failed with: ", filename)
		#df = pd.read_csv(sim[i], delimiter=',', names=list(range(156))) # handle the few rows of different lengths; most are 150

	return


data_path = '/blue/sarahballard/c.lam/sculpting2/simulations2/limbach-hybrid/'
#print("path: ", path)

# group file names by {m, b, cutoff} simulation
# note that some will be empty because I skip over them due to redundancy check from simulate_main.py
sims = []
ms = []
bs = []
cs = []
fs = []

# do first run without a done-ness check
# after first run, read in logL_incrementally_fgk.csv and get list of (m,b,c,f) that I've already done. Don't do those again.

#done = glob(path+'simulations2/limbach-hybrid/transits*')
if sys.argv[1] == 'new':
	file1 = open(path+"logLs_incremental.txt", "w") # "a" if appending, but then comment out the header and add a newline
	file1.write(path+"filename,m,b,c,f,logL,transit_multiplicity,intact_frac,disrupted_frac\n") # header

elif sys.argv[1] == 'not-new':
	# open existing file to resume collecting
	try:
		df_logLs = pd.read_csv(path+'logLs_incremental.txt')
		done_file = df_logLs.filename
	except:
		print("logLs_incremental.txt doesn't exist; run this file without this code block or the accompanying check first!")

	file1 = open(path+"logLs_incremental.txt", "a") 
	file1.write("\n") # start a new line

start = datetime.now()
#print("start: ", start)
for gi_m in range(11):
	for gi_b in range(11):
		for gi_c in range(11):
			print(gi_m, gi_b, gi_c) # so I know where I am

			sim = glob(data_path+'transits'+str(gi_m)+'_'+str(gi_b)+'_'+str(gi_c)+'_'+'*')
			cube = prior_grid_logslope(cube, ndim, nparams, gi_m, gi_b, gi_c)

			# cycle through different fractions of systems with planets
			for f in np.linspace(0, 1, 11):
				m = cube[0]
				b = cube[1]
				c = cube[2]
				#fs.append(f)

				for i in range(len(sim)):
					#df = pd.read_csv(sim[i], delimiter=',', names=list(range(150))) # handle the few rows of different lengths; most are 150
					#new_header = df.iloc[0] #grab the first row for the header
					#df = df[1:] #take the data less the header row
					#df.columns = new_header #set the header row as the df header

					if sys.argv[1]=='new':
						main(sim[i])

					elif sys.argv[1]=='not-new':
						if sim[i] in done_file:
							df_logL = df_logLs.loc[df_logLs.filename==sim[i]]
							if len(df_logL)<11:
								main(sim[i])
							else:
								pass
						else:
							main(sim[i])

file1.close()
