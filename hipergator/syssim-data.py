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

#path = '/Users/chris/Desktop/sculpting/syssim-tests/'
path = '/Users/chrislam/Desktop/sculpting/syssim-tests/' # new computer has different username

phys_cat = pd.read_csv(path+'physical_catalogs/physical_catalog1.csv', 
                       header=0, sep=',', skiprows=26) # first 26 lines are column descriptors
print(phys_cat)
print(phys_cat.groupby('target_id').size().reset_index(name='counts').groupby('counts').size())

obs_cat1 = pd.read_csv(path+'observed_catalogs/observed_catalog1.csv', 
                       header=0, sep=',', skiprows=26) # first 26 lines are column descriptors
print(obs_cat1)
print(obs_cat1.groupby('target_id').size().reset_index(name='counts').groupby('counts').size())

def calculate_transit_duration(P, r_star, r_planet, b, a, inc, e, omega, angle_flag): # Winn 2011s Eqn 14 & 16
    #print("take 1: ", r_planet/r_star)
    #print("take 2: ", (1+(r_planet/r_star))**2 - b**2)
    
    arg1 = np.sqrt((1+(r_planet/r_star))**2 - b**2)
    if angle_flag==True:
        arg2 = (r_star / a) * (arg1 / np.sin(np.pi/2 - inc)) # account for us being indexed at 0 rather than pi/2
    elif angle_flag==False:
        arg2 = (r_star / a) * (arg1 / np.sin(inc)) # assuming indexed at pi/2
    arg3 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16 of Winn 2011
    #print("Winn args: ", arg1, arg2, arg3)
    
    return (P / np.pi) * np.arcsin(arg2) * arg3

def calculate_transit_duration_paw(P, star_radius, planet_radius, b, a, incl, e, omega): # PAW website
    arg1 = np.sqrt((star_radius+planet_radius)**2 - (b*star_radius)**2) 
    arg2 = arg1/a
    arg3 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    #print("PAW args: ", arg1, arg2, arg3)
    return (P / np.pi) * np.arcsin(arg2) * arg3

def calculate_transit_duration_he(P, star_radius, planet_radius, a):
    arg1 = (P/np.pi) * (star_radius/a)
    arg2 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    arg2 = 1
    return arg1 * arg2

def calculate_sn(P, rp, rs, cdpp, tdur, unit_test_flag=False): 
    """
    Calculate S/N per planet using Eqn 4 in Christiansen et al 2012: https://arxiv.org/pdf/1208.0595.pdf
    
    Params: P (days); rp (Earth radii); rs (Solar radii); cdpp (ppm); tdur (days)
    
    Returns: S/N
    """
    tobs = 365*3.5 # days; time spanned observing the target; set to 3.5 years, or the length of Kepler mission
    f0 = 0.92 # fraction of time spent actually observing and not doing spacecraft things
    tcdpp = 0.25 # days; using CDPP for 6 hour transit durations; could change to be more like Earth transiting Sun?
    rp = solar_radius_to_au(rp) # earth_radius_to_au when not using Matthias's test set
    rs = solar_radius_to_au(rs)
    #print(P, rp, rs, cdpp, tdur)
    
    factor1 = np.sqrt(tobs*f0/np.array(P)) # this is the number of transits
    delta = 1e6*(rp/rs)**2 # convert from parts per unit to ppm
    cdpp_eff = cdpp * np.sqrt(tcdpp/tdur)
    #print("CDPP ingredients: ", cdpp, tcdpp, tdur)
    factor2 = delta/cdpp_eff
    sn = factor1 * factor2
    
    if unit_test_flag==True:
        if np.isnan(sn)==True:
            sn = 0
        return sn
    else:
        sn = sn.fillna(0)
        return sn

def calculate_transit_unit_test(planet_radius, star_radius, P, e, incl, omega, star_mass, cdpp):
    """
    accepts columns of the physical_catalogN dataframe
    """
    
    # reformulate P as a in AU
    a = p_to_a(P, star_mass)
    print("a: ", a)
    
    # calculate impact parameters; distance units in solar radii
    b = calculate_impact_parameter(star_radius, a, e, incl, omega, angle_flag=False)
    print("b: ", b)
    
    # calculate transit durations using Winn 2011 formula; same units as period
    tdur = calculate_transit_duration(P, solar_radius_to_au(star_radius), 
                            solar_radius_to_au(planet_radius), b, a, incl, e, omega, angle_flag=False)
    
    tdur_paw = calculate_transit_duration_paw(P, solar_radius_to_au(star_radius), 
                            solar_radius_to_au(planet_radius), b, a, incl, e, omega)
    
    tdur_he = calculate_transit_duration_he(P, solar_radius_to_au(star_radius),
                                           solar_radius_to_au(planet_radius), a)
    print("transit durations: ", tdur, tdur_paw, tdur_he)
    
    # calculate SN based on Eqn 4 in Christiansen et al 2012
    sn = calculate_sn(P, planet_radius, star_radius, cdpp, tdur_he, unit_test_flag=True)
    print("sn: ", sn)
    
    prob_detection = 0.1*(sn-6)
    if prob_detection < 0:
        prob_detection = 0
    elif prob_detection > 0:
        prob_detection = 1
    
    print("prob detection: ", prob_detection)
    transit_status = np.random.choice([1,0], p=[prob_detection, 1-prob_detection])
    
    """
    # calculate Fressin detection probability based on S/N
    #ts2 = [1 if sn_elt >= 7.1 else 0 for sn_elt in sn] # S/N threshold before Fressin et al 2013
    prob_detection = np.array([0.1*(sn_elt-6) for sn_elt in sn]) # S/N threshold using Fressin et al 2013
    prob_detection[np.isnan(prob_detection)] = 0 # replace NaNs with zeros
    prob_detection = np.where(prob_detection < 0., 0., prob_detection) # replace negative probs with zeros
    # actually, replace all probabilities under 5% with 5% to avoid over-penalizing models which terminate at 0% too early
    prob_detection = np.where(prob_detection > 1, 1, prob_detection) # replace probs > 1 with just 1
    prob_detections.append(prob_detection)
    
    # sample transit status and multiplicity based on Fressin detection probability
    #transit_status = [ts1_elt * ts2_elt for ts1_elt, ts2_elt in zip(ts1, ts2)]
    transit_status = [np.random.choice([1, 0], p=[pd, 1-pd]) for pd in prob_detection]
    transit_statuses.append(transit_status)
    transit_multiplicities.append(len([ts for ts in transit_status if ts == 1]))
    #transit_multiplicities.append(len([param for param in b if np.abs(param) <= 1.]))
    """
    
    return transit_status

def calculate_transit_array(star_radius, P, e, incl, omega, star_mass, planet_radius, planet_mass, cdpps):
    """
    accepts columns of the physical_catalogN dataframe
    """
    
    prob_detections = []
    transit_statuses = []
    transit_multiplicities = []
    
    # reformulate P as a in AU
    a = p_to_a(P, star_mass)
    
    # calculate impact parameters; distance units in solar radii
    b = calculate_impact_parameter(star_radius, a, e, incl, omega, angle_flag=False)
    
    # make sure arrays have explicitly float elements
    planet_radius = planet_radius.astype(float)
    star_radius = star_radius.astype(float)
    b = b.astype(float)
    a = a.astype(float)
    incl = incl.astype(float)
    e = e.astype(float)
    omega = omega.astype(float)
    
    # calculate transit durations using Winn 2011 formula; same units as period
    #tdur = calculate_transit_duration(P, solar_radius_to_au(star_radius), 
    #                        earth_radius_to_au(planet_radius), b, a, incl, e, omega)
    # Matthias's planet params are in solar units
    tdur = calculate_transit_duration(P, solar_radius_to_au(star_radius), 
                            solar_radius_to_au(planet_radius), b, a, incl, e, omega, angle_flag=False)
    
    tdur_paw = calculate_transit_duration_paw(P, solar_radius_to_au(star_radius), 
                            solar_radius_to_au(planet_radius), b, a, incl, e, omega)
    
    tdur_he = calculate_transit_duration_he(P, solar_radius_to_au(star_radius),
                                           solar_radius_to_au(planet_radius), a)
    print("transit durations: ", tdur, tdur_paw, tdur_he)
    
    # calculate CDPP by drawing from Kepler dataset relation with star radius
    #cdpp = [draw_cdpp(sr, berger_kepler) for sr in star_radius]
    
    # calculate SN based on Eqn 4 in Christiansen et al 2012
    sn = calculate_sn(P, planet_radius, star_radius, cdpps, tdur, unit_test_flag=False)
    print("number of nonzero SN: ", len(np.where(sn>0)[0]))
    
    # calculate Fressin detection probability based on S/N
    #ts2 = [1 if sn_elt >= 7.1 else 0 for sn_elt in sn] # S/N threshold before Fressin et al 2013
    #prob_detection = np.array([0.1*(sn_elt-6) for sn_elt in sn]) # S/N threshold using Fressin et al 2013
    #prob_detection[np.isnan(prob_detection)] = 0 # replace NaNs with zeros
    prob_detection = 0.1*(sn-6) # vectorize
    prob_detection = np.where(prob_detection < 0., 0., prob_detection) # replace negative probs with zeros
    # actually, replace all probabilities under 5% with 5% to avoid over-penalizing models which terminate at 0% too early
    prob_detection = np.where(prob_detection > 1, 1, prob_detection) # replace probs > 1 with just 1
    prob_detections.append(prob_detection)
    
    # sample transit status and multiplicity based on Fressin detection probability
    #transit_status = [ts1_elt * ts2_elt for ts1_elt, ts2_elt in zip(ts1, ts2)]
    transit_status = [np.random.choice([1, 0], p=[pd, 1-pd]) for pd in prob_detection]
    transit_statuses.append(transit_status)
    transit_multiplicities.append(len([ts for ts in transit_status if ts == 1]))
    #transit_multiplicities.append(len([param for param in b if np.abs(param) <= 1.]))
    
    return prob_detections, transit_statuses, transit_multiplicities, sn

def draw_cdpp(star_radius, df):
	### draw CDPP from a star in the Berger Kepler sample that's within 0.3 Solar radii from the given stellar radius
	### wait, what happens if I'm given a star that's a different size? I don't think that's a problem here.
	### also, what about Teff? Now, that would be a problem bc Matthias has M dwarfs and I don't.
    df = df.loc[(df.iso_rad<star_radius+0.15)&(df.iso_rad>star_radius-0.15)]
    #print("df length: ", len(df))
    try: 
        cdpp = np.random.choice(df.rrmscdpp06p0)
    except: # if star is not in the sizr range of Berger stars
        cdpp = np.nan

    return cdpp

def draw_cdpp_array(star_radius, df):
    # calculate CDPP by drawing from Kepler dataset relation with star radius
    cdpp = [draw_cdpp(sr, df) for sr in star_radius]
    return cdpp

### introduce data for CDPP draws ###
path = '/Users/chrislam/Desktop/sculpting/' # again, change filepath as appropriate
berger_kepler = pd.read_csv(path+'berger_kepler_stellar17.csv') # crossmatched with Gaia via Bedell
#print(list(berger_kepler.columns))

### pass phys_cat through my own detection pipeline and see how it matches obs_cat1
index = 154
planet_radius = phys_cat.planet_radius[index]
star_radius = phys_cat.star_radius[index]
P = phys_cat.period[index] # worry about errors later
e = phys_cat.ecc[index]
incl = phys_cat.incl[index]
omega = phys_cat.omega[index]
star_mass = phys_cat.star_mass[index]
cdpp = draw_cdpp(star_radius, berger_kepler)

#print("star properties: ", planet_radius, star_radius, P, e, incl, omega, star_mass, cdpp)

transit_status = calculate_transit_unit_test(planet_radius=planet_radius, star_radius=star_radius, P=P, e=e, incl=incl, omega=omega, 
                  star_mass=star_mass, cdpp=cdpp)
#print(transit_status)
#quit()

cdpps = draw_cdpp_array(phys_cat.star_radius, berger_kepler)
print("CDPPs: ", np.nanmedian(cdpps), np.nanmean(cdpps))

prob_detections, transit_statuses, transit_multiplicities, sn = calculate_transit_array(star_radius=phys_cat.star_radius, 
                        P=phys_cat.period, e=phys_cat.ecc, incl=phys_cat.incl, 
                        omega=phys_cat.omega, star_mass=phys_cat.star_mass, 
                        planet_radius = phys_cat.planet_radius, planet_mass = phys_cat.planet_mass,
                        cdpps = cdpps)

"""
prob_detections, transit_statuses, transit_multiplicities, sn = calculate_transit_vectorized(phys_cat.period, phys_cat.star_radius, 
                        solar_radius_to_earth_radius(phys_cat.planet_radius), phys_cat.ecc, phys_cat.incl, 
                        phys_cat.omega, phys_cat.star_mass, 
                        cdpps, False)
"""
#print(transit_statuses)

phys_cat['transit_status'] = transit_statuses[0]
print(phys_cat)

transiters = phys_cat.loc[phys_cat['transit_status']==1]
print(transiters.groupby('target_id').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index())

#transit_multiplicity = list(frac*transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid)
