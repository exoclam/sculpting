##################################################
### Helper functions only ########################
##################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma

path = '/blue/sarahballard/c.lam/sculpting/'
#path = '/Users/chrislam/Desktop/sculpting/' # new computer has different username

### helper conversion functions
def p_to_a(P, star_mass):
    """
    Newton's full version of Kepler's Third Law, assuming planet mass m --> 0
    Params: 
    - P: days
    - star_mass: Solar masses
    """

    P = P*86400 # days to seconds
    star_mass = star_mass*1.989e30 # solar mass to kg
    a_in_meters = (((P**2) * 6.67e-11 * star_mass)/(4*np.pi**2))**(1./3) # SI units(!)
    a = a_in_meters/(1.496e11) # meters to AU
    return a # in AU

def solar_radius_to_au(radius):
    return 0.00465047*radius

def earth_radius_to_au(radius):
    return 4.26352e-5*radius

### helper main functions
def compute_prob(x, m, b, cutoff): # adapted from Ballard et al in prep, log version
    # calculate probability of intact vs disrupted
    x = x*1e9
    if x <= 1e8: # we don't care about (nor do we have) systems before 1e8 years
        y = b

    elif (x > 1e8) & (x <= cutoff): # pre-cutoff regime
        #print(np.log10(x_elt), m, b)
        y = b + m*(np.log10(x)-8)

    elif x > cutoff: # if star is older than cutoff, use P(intact) at cutoff time
        y = b + m*(np.log10(cutoff)-8)

    if y < 0: # handle negative probabilities
        y = 0
    elif y > 1:
        y = 1
            
    return y

### helper physical transit functions
def calculate_eccentricity_limbach(multiplicity):
    """
    Draw eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    Params: multiplicity of system (int)
    Returns: np.array of eccentricity values with length==multiplicity
    """
    # for drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    limbach = pd.read_csv(path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator

    values = np.random.rand(multiplicity) # draw an eccentricity per planet
    if multiplicity==1:
        value_bins = np.searchsorted(limbach['1'], values) # return positions in cdf vector where random values should go
    elif multiplicity==2:
        value_bins = np.searchsorted(limbach['2'], values) # return positions in cdf vector where random values should go
    elif multiplicity==5:
        value_bins = np.searchsorted(limbach['5'], values) # return positions in cdf vector where random values should go
    elif multiplicity==6:
        value_bins = np.searchsorted(limbach['6'], values) # return positions in cdf vector where random values should go
    random_from_cdf = np.logspace(-2,0,101)[value_bins] # select x_d positions based on these random positions
    
    return random_from_cdf

def draw_eccentricity_van_eylen(model_flag, num_planets):
    """
    Draw eccentricities per the four models of Van Eylen et al 2018 (https://arxiv.org/pdf/1807.00549.pdf)
    Params: flag (string) saying which of the four models; num_planets (int)
    Returns: list eccentricity per planet in the system
    """
    if model_flag=='rayleigh':
        sigma_single = 0.24
        sigma_multi = 0.061
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi
            
        draw = np.random.rayleigh(sigma, num_planets)

    elif model_flag=='half-Gaussian':
        sigma_single = 0.32
        sigma_multi = 0.083
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi
            
        draw = np.random.normal(0, sigma, num_planets)
        if any(d < 0 for d in draw): # put the 'half' in half-Gaussian by redrawing if any draw element is negative
            draw = draw_eccentricity_van_eylen('half-Gaussian', num_planets)
        
    elif model_flag=='beta':
        a_single = 1.58
        b_single = 4.4
        a_multi = 1.52
        b_multi = 29
        
        # errors for pymc3 implementation of eccentricity draws, should I wish to get fancy.
        # I currently do not wish to get fancy.
        a_single_err1 = 0.59
        a_single_err2 = 0.93
        b_single_err1 = 1.8
        b_single_err2 = 2.2
        a_multi_err1 = 0.5
        a_multi_err2 = 0.85
        b_multi_err1 = 9
        b_multi_err2 = 17
        
        if num_planets==1:
            a = a_single
            b = b_single
        elif num_planets>1:
            a = a_multi
            b = b_multi
        
        draw = np.random.beta(a, b, num_planets)
        
    elif model_flag=='mixed':
        sigma_half_gauss = 0.049
        sigma_rayleigh = 0.26
        f_single = 0.76
        f_multi = 0.08
        
        if num_planets==1:
            draw = np.random.rayleigh(sigma_rayleigh, num_planets)
        elif num_planets>1:
            draw = np.random.normal(0, sigma_half_gauss, num_planets)
            if any(d < 0 for d in draw): # redraw if any planet's eccentricity is negative
                draw = draw_eccentricity_van_eylen('mixed', num_planets)
                
    elif model_flag=='mixed-limbach':
        """
        Testing something for Sarah: use Rayleigh for intact and Limbach for disrupted. 
        """
        sigma_rayleigh = 0.26
        #print(num_planets)
        if num_planets==1:
            draw = np.random.rayleigh(sigma_rayleigh, num_planets)
        elif num_planets>1:
            draw = calculate_eccentricity_limbach(num_planets)
            
    elif model_flag=='limbach': # OG way of drawing eccentricities, from Limbach & Turner 2014
        draw = calculate_eccentricity_limbach(num_planets)
            
    return draw

def calculate_impact_parameter(star_radius, a, e, incl, omega, angle_flag): # following Winn 2010 Eqn 7
    """
    angle_flag: True means indexed at 0; False means indexed at pi/2
    """
    star_radius = solar_radius_to_au(star_radius)
    if angle_flag==True:
        factor1 = (a * np.cos(np.pi/2 - incl))/star_radius  # again, we're indexed at 0 rather than pi/2
    elif angle_flag==False: # if indexed at pi/2
        factor1 = (a * np.cos(incl))/star_radius 
    factor2 = (1-e**2)/(1+e*np.sin(omega)) # leave this alone, right? Bc everyone always assumes omega=pi/2?
    
    return factor1 * factor2

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

def calculate_transit_duration_paw(P, star_radius, planet_radius, b, a, incl, e, omega): # Paul Anthony Wilson website: https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    arg1 = np.sqrt((star_radius+planet_radius)**2 - (b*star_radius)**2) 
    arg2 = arg1/a
    arg3 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    #print("PAW args: ", arg1, arg2, arg3)
    return (P / np.pi) * np.arcsin(arg2) * arg3

def calculate_transit_duration_he(P, star_radius, planet_radius, a, e, omega): # from Matthias He: https://github.com/ExoJulia/SysSimExClusters/tree/master/src
    arg1 = (P/np.pi) * (star_radius/a)
    arg2 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    arg2 = 1
    return arg1 * arg2

### helper transit detection functions
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

def draw_cdpp(star_radius, df):
    df = df.loc[(df.st_radius<star_radius+0.15)&(df.st_radius>star_radius-0.15)]
    #print("df length: ", len(df))
    cdpp = np.random.choice(df.rrmscdpp06p0)
    return cdpp

def draw_cdpp_array(star_radius, df):
    # calculate CDPP by drawing from Kepler dataset relation with star radius
    cdpp = [draw_cdpp(sr, berger_kepler) for sr in star_radius]
    return cdpp