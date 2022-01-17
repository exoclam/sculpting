##################################################
### Functions for the physical pipeline ##########
##################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma
from simulate_helpers import *
import matplotlib.pyplot as plt

def calculate_transit_unit_test(planet_radius, star_radius, P, e, incl, omega, star_mass, cdpp):
    
    # reformulate P as a in AU
    a = p_to_a(P, star_mass)
    print("a: ", a)
    
    # calculate impact parameters; distance units in solar radii
    b = calculate_impact_parameter(star_radius, a, e, incl, omega, angle_flag=False)
    print("b: ", b)
    
    # calculate transit durations using Winn 2011 formula; same units as period
    tdur = calculate_transit_duration(P, solar_radius_to_au(star_radius), 
                            earth_radius_to_au(planet_radius), b, a, incl, e, omega, angle_flag=False)
    
    tdur_paw = calculate_transit_duration_paw(P, solar_radius_to_au(star_radius), 
                            earth_radius_to_au(planet_radius), b, a, incl, e, omega)
    
    tdur_he = calculate_transit_duration_he(P, solar_radius_to_au(star_radius),
                                           earth_radius_to_au(planet_radius), a, e, omega)
    ###print("transit durations: ", tdur, tdur_paw, tdur_he)
    
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

def calculate_transit_array(star_radius, P, e, incl, omega, star_mass, planet_radius, cdpps):
    """
    This was used prior to switching to HiPerGator. Still useful for testing in Jupyter.
    Params: 
    - star_radius: in Solar radii
    - P: period in days
    - e: eccentricity
    - incl: inclination
    - omega: longitude of periastron
    - star_mass: in Solar masses
	- planet_radius: in Earth radii
	- cdpps: Combined Differential Photometric Precision, a measure of stellar noise
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
    tdur = calculate_transit_duration(P, solar_radius_to_au(star_radius), 
                            earth_radius_to_au(planet_radius), b, a, incl, e, omega, angle_flag=False)
    
    tdur_paw = calculate_transit_duration_paw(P, solar_radius_to_au(star_radius), 
                            earth_radius_to_au(planet_radius), b, a, incl, e, omega)
    
    tdur_he = calculate_transit_duration_he(P, solar_radius_to_au(star_radius),
                                           earth_radius_to_au(planet_radius), a, e, omega)
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

def calculate_transit_me_with_amd(P, star_radius, planet_radius, e, incl, omega, star_mass, cdpps, angle_flag):
    """
    Params: columns of the berger_kepler dataframe
    Returns:
    - Probabilities of detection: Numpy array
    - Transit statuses: Numpy array
    - Transit multiplicities (lambdas for calculating logLs): Numpy array
    - S/N ratios: Numpy array
    """

    #print(P, star_radius, planet_radius, e, incl, omega, star_mass, cdpps)
    prob_detections = []
    transit_statuses = []
    transit_multiplicities = []
    #planet_radius = 2.       
    
    # reformulate P as a in AU
    a = p_to_a(P, star_mass)
    
    # calculate impact parameters; distance units in solar radii
    b = calculate_impact_parameter(star_radius, a, e, incl, omega, angle_flag)
    
    # make sure arrays have explicitly float elements
    planet_radius = planet_radius.astype(float)
    star_radius = star_radius.astype(float)
    b = b.astype(float)
    a = a.astype(float)
    incl = incl.astype(float)
    e = e.astype(float)
    omega = omega.astype(float)
    cdpps = cdpps.astype(float)
    P = P.astype(float)
    ###print("b: ", b)
    # calculate transit durations using Winn 2011 formula; same units as period
    #tdur = calculate_transit_duration(P, solar_radius_to_au(star_radius), 
    #                        earth_radius_to_au(planet_radius), b, a, incl, e, omega)
    # Matthias's planet params are in solar units
    tdur = calculate_transit_duration(P, solar_radius_to_au(star_radius), 
                            earth_radius_to_au(planet_radius), b, a, incl, e, omega, angle_flag)
    ###print("tdur: ", tdur)
    tdur_paw = calculate_transit_duration_paw(P, solar_radius_to_au(star_radius), 
                            earth_radius_to_au(planet_radius), b, a, incl, e, omega)
    
    tdur_he = calculate_transit_duration_he(P, solar_radius_to_au(star_radius),
                                           earth_radius_to_au(planet_radius), a, e, omega)
    #print("transit durations: ", tdur, tdur_paw, tdur_he)
    
    # calculate CDPP by drawing from Kepler dataset relation with star radius
    #cdpp = [draw_cdpp(sr, berger_kepler) for sr in star_radius]
    
    # calculate SN based on Eqn 4 in Christiansen et al 2012
    sn = calculate_sn(P, planet_radius, star_radius, cdpps, tdur, unit_test_flag=False)
    ###print("number of nonzero SN: ", len(np.where(sn>0)[0]))
    
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

def model_direct_draw(cube):
    """
    This is the old model. See model_van_eylen().
    Take prior and draw systems out of Kepler data 
    Params: n-dim hypercube prior
    Returns: lambdas and simulated transit DataFrame
    """
    # retrieve prior
    #print(cube)
    m, b, cutoff = cube[0], cube[1], cube[2]
    
    kepids = []
    midplanes = []
    midplanes_degrees = []
    inclinations = []
    inclinations_degrees = []
    eccentricities = []
    long_periastrons = []
    planets_per_case2 = [] # maybe better to recreate it here b/c I can then generalize it for Case 2?
    planets_a_case2 = []
    star_radius = []
    star_mass = []
    impact_parameters = []
    transit_status1 = []
    transit_status2 = []
    transit_statuses = []
    transit_multiplicities = []
    geometric_transit_multiplicity = []
    tdurs = []
    jmags = []
    sns = []
    prob_detections = []
    xi = []
    xi_old = []
    xi_young = []
    prob_intacts = []
    amds = []
    num_planets_all = []
    intacts = 0
    
    # draw ~20000 systems
    num_samples = len(berger_kepler)
    #num_samples = 100
    for i in range(len(berger_kepler[0:num_samples])):
    #for i in range(10):
        ### star ###
        kepids.append(berger_kepler.kepid[i])
        r_star = berger_kepler.iso_rad[i] # in solar radii
        m_star = berger_kepler.iso_mass[i] # in solar masses
        age_star = berger_kepler.iso_age[i]
        mu = list(np.random.uniform(-np.pi/2,np.pi/2,1)) # create midplane for star 
        midplanes.append(mu)
        midplanes_degrees.append([mu_elt*180/np.pi for mu_elt in mu])
        cdpp = berger_kepler.rrmscdpp06p0[i] # don't convert from ppm to parts per unit

        ### planet ###
        r_planet = 2. # use two Earth radii; will make negligible difference
        m_planet = 5. # from Chen & Kipping 2016
        
        """
        # calculate probability given age using piecewise model
        #print("m, b, cutoff: ", m, b, cutoff)
        #print("age star: ", age_star)
        if age_star < cutoff: # if decay cutoff hasn't occurred yet, follow decay model
            prob = m * age_star + b
        else: # if decay cutoff has occurred, use P(intact) at time when cutoff occurred
            prob = m * cutoff + b 
        """

        """
        # not including piecewise/cutoff model
        prob = m * age_star + b

        if prob < 0.: # don't allow negative probabilities
            prob = 0.
        """

        prob = compute_prob(age_star, m, b, cutoff)
        prob_intacts.append(prob)
        intact_flag = np.random.choice(['intact', 'disrupted'], p=[prob, 1-prob])
        if intact_flag == 'intact':
            intacts += 1
            # young system has 5 or 6 planets
            num_planets = random.choice([5, 6]) 
            sigma = np.pi/90 # 2 degrees, per Fig 6 in Fabrycky 2012
            eccentricity = calculate_eccentricity_limbach(num_planets)
            long_periastron = np.random.uniform(0,2*np.pi,1) # or do I draw num_planet omegas?
            
            # simulate transit-related characteristics for 5 or 6 planets
            planet_a_case2, inclination_degrees = sim_transits_new(r_star, m_star, num_planets, mu, sigma, r_planet, age_star, eccentricity, long_periastron,
                             planets_per_case2 = planets_per_case2, planets_a_case2 = planets_a_case2, 
                             inclinations = inclinations, inclinations_degrees = inclinations_degrees,
                             impact_parameters = impact_parameters, transit_statuses = transit_statuses, 
                             transit_status1 = transit_status1, transit_status2 = transit_status2, 
                             transit_multiplicities = transit_multiplicities, tdurs = tdurs,
                             cdpp = cdpp, sns = sns, prob_detections = prob_detections, 
                             geometric_transit_multiplicity = geometric_transit_multiplicity)

        elif intact_flag == 'disrupted':
            # old system has 1 or 2 planets
            num_planets = random.choice([1, 2]) 
            sigma = np.pi/22.5 # 8 degree disk plane inclination scatter
            eccentricity = calculate_eccentricity_limbach(num_planets)
            long_periastron = np.random.uniform(0,2*np.pi,1) # or do I draw num_planet omegas?
            
            # simulate transit-related characteristics for 1 or 2 planets
            planet_a_case2, inclination_degrees = sim_transits_new(r_star, m_star, num_planets, mu, sigma, r_planet, age_star, eccentricity, long_periastron,
                             planets_per_case2 = planets_per_case2, planets_a_case2 = planets_a_case2, 
                             inclinations = inclinations, inclinations_degrees = inclinations_degrees, 
                             impact_parameters = impact_parameters, transit_statuses = transit_statuses, 
                             transit_status1 = transit_status1, transit_status2 = transit_status2,
                             transit_multiplicities = transit_multiplicities, tdurs = tdurs,
                             cdpp = cdpp, sns = sns, prob_detections = prob_detections,
                             geometric_transit_multiplicity = geometric_transit_multiplicity)

        num_planets_all.append(num_planets)
        eccentricities.append(eccentricity)
        long_periastrons.append(long_periastron)
        
        # calculate AMD per system
        amd = calculate_amd(earth_mass_to_cgs(m_planet), solar_mass_to_cgs(m_star), 
                        planet_a_case2, eccentricity, inclination_degrees, num_planets)
        amds.append(amd)
        
    # calculate AMD per system
    #amds = calculate_amd(earth_mass_to_cgs(m_planet), solar_mass_to_cgs(berger_kepler.iso_mass[0:num_samples]), 
    #                    planets_a_case2, eccentricities, inclinations, num_planets)

    midplanes = np.concatenate(midplanes, axis=0) # turn list of lists of one into regular list
    intact_fractions = intacts/num_samples
    #print(intacts, num_samples, intact_fractions)
    
    """
    transits_dict = {'star_ages': berger_kepler.iso_age, 'planet_periods': planets_per_case2, 
    'semi_major_axes': planets_a_case2, 'midplane': midplanes, 'midplane_degrees': midplanes_degrees,
                     'planet_inclinations': inclinations, 'planet_inclinations_degrees': inclinations_degrees,
                     'impact_parameters': impact_parameters, 'transit_status': transit_statuses, 
                     'transit_multiplicity': transit_multiplicities, 'kepid': kepids,
                     'y_intercept': b, 'slope': m, 'transit_duration': tdurs, 
                     '6hr_cdpp': berger_kepler.rrmscdpp06p0, 'signal_noise': sns,
                     'prob_detections': prob_detections}
    """
    transits_dict = {'star_ages': berger_kepler.iso_age[0:num_samples], 'planet_periods': planets_per_case2[0:num_samples], 
    'semi_major_axes': planets_a_case2[0:num_samples], 'midplane': midplanes[0:num_samples], 'midplane_degrees': midplanes_degrees[0:num_samples],
                     'planet_inclinations': inclinations[0:num_samples], 'planet_inclinations_degrees': inclinations_degrees[0:num_samples],
                     'eccentricities': eccentricities[0:num_samples], 'amds': amds[0:num_samples], 'long_periastons': long_periastrons[0:num_samples],
                     'impact_parameters': impact_parameters[0:num_samples], 'transit_status': transit_statuses[0:num_samples], 
                     'geometric_transit': transit_status1[0:num_samples], 'geometric_transit_multiplicity': geometric_transit_multiplicity[0:num_samples],
                     'transit_multiplicity': transit_multiplicities[0:num_samples], 'kepid': kepids[0:num_samples],
                     'y_intercept': b*np.ones(num_samples), 'slope': m*np.ones(num_samples), 'transit_duration': tdurs[0:num_samples], 
                     '6hr_cdpp': berger_kepler.rrmscdpp06p0[0:num_samples], 'signal_noise': sns[0:num_samples],
                     'prob_detections': prob_detections[0:num_samples], 'prob_intacts': prob_intacts[0:num_samples],
                    'num_planets': num_planets_all[0:num_samples]}
    
    length_dict = {key: len(value) for key, value in transits_dict.items()}
    #print(length_dict)
    transits = pd.DataFrame(transits_dict)    
    
    #lam = transits.transit_multiplicity.value_counts()
    #lam = transits.loc[transits.transit_multiplicity > 0].transit_multiplicity.value_counts() * (np.sum(k_old)/len(transits.loc[transits.transit_multiplicity > 0]))
    #lam = transits.loc[transits.transit_multiplicity > 0].transit_multiplicity.value_counts() * (len(berger_kepler)/num_samples) # scale up to full counts of k
    lam = transits.transit_multiplicity.value_counts().reindex(transits.index[0:6], # to deal w/zero value gaps 
                                                               fill_value=0) * (len(berger_kepler)/num_samples)
    geom_lam = transits.geometric_transit_multiplicity.value_counts().reindex(transits.index[0:6], # to deal w/zero value gaps 
                                                               fill_value=0) * (len(berger_kepler)/num_samples)

    lam = lam.to_list()
    geom_lam = geom_lam.to_list()
    return lam, geom_lam, transits, intact_fractions, amds, eccentricities, inclinations_degrees

def model_van_eylen(star_age, df, model_flag, cube):
    """
    Enrich berger_kepler DataFrame with planet parameters like ecc, incl, etc.
    Params: 
    - star_age: berger_kepler.iso_age (Pandas Series of floats)
    - df: berger_kepler (DataFrame of star and planet params)
    - model_flag: what initial eccentricity distribution to use (string)
    - cube: list of [m, b, cutoff in years, fraction of systems with planets]
    - Just kidding, cube will just be [m, b, cutoff]...frac will be used in likelihood_main.py
    Returns:
    - Pandas DataFrame: an enriched version of df
    """
    
    # unpack model parameters from hypercube
    m, b, cutoff = cube[0], cube[1], cube[2]

    periods = [] # the column upon which we'll explode berger_kepler to make berger_kepler_planets
    num_planets_all = [] # intrinsic planets
    eccs = []
    incls = []
    omegas = []
    intacts = 0
    midplanes = []
    intact_flags = []
    mutual_incls = []

    for age in star_age:
        # sometimes make more than one planet per system
        prob = compute_prob(age, m, b, cutoff)
        #print(age, prob)
        
        # midplane
        midplane = np.random.uniform(-np.pi/2,np.pi/2,1)
        midplanes.append(midplane)

        intact_flag = np.random.choice(['intact', 'disrupted'], p=[prob, 1-prob])
        intact_flags.append(intact_flag)
        if intact_flag == 'intact':
            intacts += 1
            # young system has 5 or 6 planets
            num_planets = random.choice([5, 6]) 
            
            # for drawing incl
            sigma = np.pi/90 # 2 degrees, per Fig 6 in Fabrycky 2012
            
        elif intact_flag == 'disrupted':
            # old system has 1 or 2 planets
            num_planets = random.choice([1, 2]) 

            # for drawing incl
            sigma = np.pi/22.5 # 8 degrees, per Fig 6 in Fabrycky 2012
            
        # draw period from loguniform distribution from 2 to 300 days
        P = np.array(loguniform.rvs(2, 300, size=num_planets))
        periods.append(P)
        
        # draw inclinations from Gaussian distribution centered on midplane
        incl = np.random.normal(midplane, sigma, num_planets)
        #incl = [np.pi/2 if inc_elt > np.pi/2 else inc_elt for inc_elt in incl] # artificially impose bounds post-facto
        #incl = [-np.pi/2 if inc_elt < -np.pi/2 else inc_elt for inc_elt in incl] # lower bound
        incls.append(incl)

        # obtain mutual inclinations for plotting to compare {e, i} distributions
        mutual_incl = midplane - incl
        mutual_incls.append(mutual_incl)

        # draw eccentricity
        ecc = draw_eccentricity_van_eylen(model_flag, num_planets)
        eccs.append(ecc)

        # draw longitudes of periastron
        omega = np.random.uniform(0, 2*np.pi, num_planets)
        omegas.append(omega)
        
        num_planets_all.append(num_planets)

    """
    plt.hist(np.array(midplanes)*180/np.pi, bins=100)
    plt.savefig('midplanes.png')

    plt.hist(np.array(incls)*180/np.pi, bins=100)
    plt.savefig('incls.png')
    quit()
    """
    df['P'] = periods
    df['midplane'] = midplanes
    df['intact_flag'] = intact_flags
    df['num_planets'] = num_planets_all
    berger_kepler_planets = df.explode('P')

    berger_kepler_planets = df.explode('P', '')
    #print(berger_kepler_planets)
    ###print("intacts: ", intacts)
    
    # draw longitudes of periastron
    #omega = np.random.uniform(0,2*np.pi,len(berger_kepler_planets))
    #berger_kepler_planets['omega'] = omega
    
    eccs = np.asarray([item for ecc_elt in eccs for item in ecc_elt])
    incls = np.asarray([item for incl_elt in incls for item in incl_elt])
    mutual_incls = np.asarray([item for mutual_incl_elt in mutual_incls for item in mutual_incl_elt])
    omegas = np.asarray([item for omega_elt in omegas for item in omega_elt])
    #num_planets_all = np.asarray([item for numplanet_elt in num_planets_all for item in numplanet_elt])
    #print(len(incls), len(eccs), len(omega))
    #print(np.pi/2 - incls)
    berger_kepler_planets['ecc'] = eccs
    berger_kepler_planets['incl'] = incls
    berger_kepler_planets['omega'] = omegas
    berger_kepler_planets['planet_radius'] = 2.
    berger_kepler_planets['mutual_incl'] = mutual_incls
    #berger_kepler_planets['intact_flag'] = intact_flags
    #print(len(num_planets_all), len(omegas))
    #berger_kepler_planets['num_planets'] = num_planets_all
    ###print("ecc: ", berger_kepler_planets['ecc'])
    ###print("incl: ", berger_kepler_planets['incl'])
    ###print("mean ecc and incl: ", np.mean(berger_kepler_planets['ecc']), np.mean(berger_kepler_planets['incl']))

    prob_detections, transit_statuses, transit_multiplicities, sn = calculate_transit_me_with_amd(berger_kepler_planets.P, 
                            berger_kepler_planets.iso_rad, berger_kepler_planets.planet_radius,
                            berger_kepler_planets.ecc, 
                            berger_kepler_planets.incl, 
                            berger_kepler_planets.omega, berger_kepler_planets.iso_mass,
                            berger_kepler_planets.rrmscdpp06p0, angle_flag=True)
    
    berger_kepler_planets['transit_status'] = transit_statuses[0]
    berger_kepler_planets['prob_detections'] = prob_detections[0]
    berger_kepler_planets['sn'] = sn
    return berger_kepler_planets


