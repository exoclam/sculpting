import pandas as pd   

#path = '/blue/sarahballard/c.lam/sculpting2/'
path = '/Users/chrislam/Desktop/sculpting/'

# for drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
limbach = pd.read_csv(path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator
