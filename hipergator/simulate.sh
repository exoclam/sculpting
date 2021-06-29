#!/bin/bash
#SBATCH --job-name=simulate    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.lam@ufl.edu     # Where to send mail	
#SBATCH --ntasks=2                    # Run on a single CPU
#SBATCH --mem=2gb                     # Job memory request
#SBATCH --time=48:05:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
pwd; hostname; date

module load python

echo "Running simulate script on two CPU cores"

python /blue/sarahballard/c.lam/sculpting/simulate.py

date
