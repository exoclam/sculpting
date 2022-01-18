#!/bin/bash
#SBATCH --job-name=simulate    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.lam@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # 
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --time=88:05:00               # Time limit hrs:min:sec
#SBATCH --output=simulate_%A_%a.log   # Standard output and error log
#SBATCH --array=0-10
pwd; hostname; date
echo "$SLURM_ARRAY_TASK_ID"

module load python

echo "Running simulate job array script"

python /blue/sarahballard/c.lam/sculpting2/simulate_main.py $SLURM_ARRAY_TASK_ID

#for cutoff in 1 2 3 4 5 6 7 8 9 10
#do
#  python /sculpting/simulate.sh $cutoff
#done

date
