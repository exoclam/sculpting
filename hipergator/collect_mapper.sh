#!/bin/bash
#SBATCH --job-name=collect_mapper    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.lam@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # 
#SBATCH --mem=2gb                     # Job memory request
#SBATCH --time=80:05:00               # Time limit hrs:min:sec
#SBATCH --output=collect_%A_%a.log   # Standard output and error log
#SBATCH --array=0-10
pwd; hostname; date
echo "$SLURM_ARRAY_TASK_ID"

module load python

echo "Running collect job array script"

python /blue/sarahballard/c.lam/sculpting2/collect_simulations.py $SLURM_ARRAY_TASK_ID

date
