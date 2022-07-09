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
#k = pd.Series([833, 134, 38, 15, 5, 0])
k = pd.Series([864, 138, 38, 15, 5, 0])
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
	#print("old lam: ", lam)
	#print("new lam: ", lam)
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

def main(filename, f): # one read and write per sim[i] filename
	try:
		df = pd.read_csv(filename, delimiter=',')
		# count geometric transits
		#geom_transits = df.loc[df['geom_transit_status']==1]
		geom_transits = list(df.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid)
		geom_transits += [0.] * (7 - len(geom_transits)) # pad with zeros to make all sets size 7

		# isolate transiting planets
		transiters_berger_kepler = df.loc[df['transit_status']==1]

		# compute transit multiplicity and save off the original transit multiplicity (pre-frac)
		transit_multiplicity = list(f * transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
		transit_multiplicity += [0.] * (len(k) - len(transit_multiplicity)) # pad with zeros to match length of k

		# calculate logLs for different fracs and keep the best one
		logL = better_loglike(transit_multiplicity, k)

		# get intact and disrupted fractions (combine them later to get fraction of systems w/o planets)
		intact = df.loc[df.intact_flag=='intact']
		disrupted = df.loc[df.intact_flag=='disrupted']
		#print(df.loc[(df.intact_flag != 'intact') & (df.intact_flag != 'disrupted')].intact_flag)
		#print("intact: ", len(intact.kepid.unique()))
		#print("disrupted: ", len(disrupted.kepid.unique()))
		#print("total: " , len(df.kepid.unique()))

		intact_frac = f*len(intact)/len(df)
		disrupted_frac = f*len(disrupted)/len(df)
		intact_frac2 = len(intact.kepid.unique())/len(df.kepid.unique())
		disrupted_frac2 = len(disrupted.kepid.unique())/len(df.kepid.unique())

		# sneak out transit_multiplicity and associated logLs for intact vs disrupted
		intact_transiting = intact.loc[intact['transit_status']==1]
		disrupted_transiting = disrupted.loc[disrupted['transit_status']==1]
		try:
			intact_transit_multiplicity = f * intact_transiting.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
			intact_logL = better_loglike(intact_transit_multiplicity, k)		
		except:
			intact_transit_multiplicity = [np.nan] # placeholder that can be removed later
			intact_logL = np.nan
		try:
			disrupted_transit_multiplicity = f * disrupted_transiting.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
			disrupted_logL = better_loglike(disrupted_transit_multiplicity, k)
		except:
			disrupted_transit_multiplicity = [np.nan] # placeholder that can be removed later
			disrupted_logL = np.nan

		# sneak out transit_multiplicity and logLs for different age cuts of young vs old
		young1 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 1.]
		old1 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 1.]
		young1_transit_multiplicity = f * young1.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old1_transit_multiplicity = f * old1.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid			
		young1_logL = better_loglike(young1_transit_multiplicity, k)
		old1_logL = better_loglike(old1_transit_multiplicity, k)

		young2 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 1.5]
		old2 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 1.5] 
		young2_transit_multiplicity = f * young2.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old2_transit_multiplicity = f * old2.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young2_logL = better_loglike(young2_transit_multiplicity, k)
		old2_logL = better_loglike(old2_transit_multiplicity, k)	

		young3 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 2.]
		old3 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 2.] 
		young3_transit_multiplicity = f * young3.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old3_transit_multiplicity = f * old3.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young3_logL = better_loglike(young3_transit_multiplicity, k)
		old3_logL = better_loglike(old3_transit_multiplicity, k)

		young4 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 2.5]
		old4 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 2.5] 
		young4_transit_multiplicity = f * young4.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old4_transit_multiplicity = f * old4.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young4_logL = better_loglike(young4_transit_multiplicity, k)
		old4_logL = better_loglike(old4_transit_multiplicity, k)

		young5 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 3.]
		old5 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 3.] 
		young5_transit_multiplicity = f * young5.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old5_transit_multiplicity = f * old5.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young5_logL = better_loglike(young5_transit_multiplicity, k)
		old5_logL = better_loglike(old5_transit_multiplicity, k)

		young6 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 3.5]
		old6 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 3.5] 
		young6_transit_multiplicity = f * young6.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old6_transit_multiplicity = f * old6.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young6_logL = better_loglike(young6_transit_multiplicity, k)
		old6_logL = better_loglike(old6_transit_multiplicity, k)

		young7 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 4.]
		old7 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 4.] 
		young7_transit_multiplicity = f * young7.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old7_transit_multiplicity = f * old7.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young7_logL = better_loglike(young7_transit_multiplicity, k)
		old7_logL = better_loglike(old7_transit_multiplicity, k)

		young8 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 4.5]
		old8 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 4.5] 
		young8_transit_multiplicity = f * young8.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old8_transit_multiplicity = f * old8.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young8_logL = better_loglike(young8_transit_multiplicity, k)
		old8_logL = better_loglike(old8_transit_multiplicity, k)

		young9 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] <= 5.]
		old9 = transiters_berger_kepler.loc[transiters_berger_kepler['iso_age'] > 5.] 
		young9_transit_multiplicity = f * young9.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		old9_transit_multiplicity = f * old9.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
		young9_logL = better_loglike(young9_transit_multiplicity, k)
		old9_logL = better_loglike(old9_transit_multiplicity, k)

		# output
		#out = np.array((str(m), str(b), str(c), str(f), str(logL), str(list(transit_multiplicity)), str(intact_frac), str(disrupted_frac)))
		out = np.array((filename, m, b, c, f, logL, list(transit_multiplicity), geom_transits, intact_frac, disrupted_frac, intact_frac2, disrupted_frac2, intact_logL, list(intact_transit_multiplicity), disrupted_logL, list(disrupted_transit_multiplicity), young1_logL, list(young1_transit_multiplicity), old1_logL, list(old1_transit_multiplicity), young2_logL, list(young2_transit_multiplicity), old2_logL, list(old2_transit_multiplicity), young3_logL, list(young3_transit_multiplicity), old3_logL, list(old3_transit_multiplicity), young4_logL, list(young4_transit_multiplicity), old4_logL, list(old4_transit_multiplicity), young5_logL, list(young5_transit_multiplicity), old5_logL, list(old5_transit_multiplicity), young6_logL, list(young6_transit_multiplicity), old6_logL, list(old6_transit_multiplicity), young7_logL, list(young7_transit_multiplicity), old7_logL, list(old7_transit_multiplicity), young8_logL, list(young8_transit_multiplicity), old8_logL, list(old8_transit_multiplicity), young9_logL, list(young9_transit_multiplicity), old9_logL, list(old9_transit_multiplicity)))
		out = out.reshape(1, len(out))
		np.savetxt(file1, out, fmt='%s', delimiter='\t', newline='\n')
		#file1.write("\n")

	except Exception as e:
		print(e)
		print("failed with: ", filename)
		#df = pd.read_csv(sim[i], delimiter=',', names=list(range(156))) # handle the few rows of different lengths; most are 150

	return

#data_path = 'home/c.lam/blue/sculpting2/simulations2/limbach-hybrid/'
data_path = '/blue/sarahballard/c.lam/sculpting2/simulations2/fixed-detection3/'
#data_path = path+'hipergator/'
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
	file1 = open(path+"simulations2/fixed-detection-results/logLs_fixed_detection.txt", "w") # "a" if appending, but then comment out the header and add a newline
	file1.write("filename,m,b,c,f,logL,transit_multiplicity,geom_transit_multiplicity,intact_frac,disrupted_frac,intact_frac2,disrupted_frac2,intact_logL,intact_transit_multiplicity,disrupted_logL,disrupted_transit_multiplicity,young10_logL,young10_transit_multiplicity,old10_logL,old10_transit_multiplicity,young15_logL,young15_transit_multiplicity,old15_logL,old15_transit_multiplicity,young20_logL,young20_transit_multiplicity,old20_logL,old20_transit_multiplicity,young25_logL,young25_transit_multiplicity,old25_logL,old25_transit_multiplicity,young30_logL,young30_transit_multiplicity,old30_logL,old30_transit_multiplicity,young35_logL,young35_transit_multiplicity,old35_logL,old35_transit_multiplicity,young40_logL,young40_transit_multiplicity,old40_logL,old40_transit_multiplicity,young45_logL,young45_transit_multiplicity,old45_logL,old45_transit_multiplicity,young50_logL,young50_transit_multiplicity,old50_logL,old50_transit_multiplicity\n") # header

elif sys.argv[1] == 'not-new':
	# open existing file to resume collecting
	df_logLs = pd.read_csv(path+'logLs_incremental_corrected.txt',sep='\s+',on_bad_lines='skip')
	#print(len(df_logLs))
	#quit()
	
	try:
		df_logLs = pd.read_csv(path+'logLs_incremental_corrected.txt')
		done_file = df_logLs.filename
	except:
		print("logLs_incremental.txt doesn't exist; run this file without this code block or the accompanying check first!")
	#quit()

	file1 = open(path+"logLs_incremental_corrected.txt", "a") 
	file1.write("\n") # start a new line

start = datetime.now()
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
						main(sim[i], f)
					elif sys.argv[1]=='not-new':
						if sim[i] in done_file:
							pass
						else:
							main(sim[i], f)	
					"""			
					elif sys.argv[1]=='not-new':
						if sim[i] in done_file:
							df_logL = df_logLs.loc[df_logLs.filename==sim[i]]
							if len(df_logL)<11:
								main(sim[i])
							else:
								pass
						else:
							main(sim[i])
					"""

file1.close()
