#!/bin/bash
#SBATCH --job-name=cobayarun_July9Test_2

#SBATCH --output=runs/%x/%x_%j.out
#SBATCH --error=runs/%x/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --constraint=cpu
#SBATCH --account=mp107a
#SBATCH -q regular

module load python

conda activate soliket-test2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export COBAYA_USE_FILE_LOCKING=False

cd /global/homes/c/cpopik/CAPPIBARAS

start_time=$(date +%s)

trap '
end_time=$(date +%s)
mkdir -p "runs/${SLURM_JOB_NAME}"
echo "---- SLURM timing ----" >> "runs/${SLURM_JOB_NAME}/runtime.txt"
echo "Started: $(date -d @$start_time)" >> "runs/${SLURM_JOB_NAME}/runtime.txt"
echo "Ended: $(date)" >> "runs/${SLURM_JOB_NAME}/runtime.txt"
echo "Wall time: $((end_time-start_time)) seconds" >> "runs/${SLURM_JOB_NAME}/runtime.txt"
' EXIT

python runchains.py