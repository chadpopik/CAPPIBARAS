#!/bin/bash
#SBATCH --job-name=cobayarun_July9Test_2

# NOTE: Slurm parses #SBATCH directives literally before any script runs, so
# these paths can't be read from config.py. Keep them in sync with OUTPUT_PATH there.
#SBATCH --output=/pscratch/sd/c/cpopik/CAPPIBARAS/runs/%x/%x_%j.out
#SBATCH --error=/pscratch/sd/c/cpopik/CAPPIBARAS/runs/%x/%x_%j.err

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

# Record time spent waiting in the SLURM queue (Submit -> Start).
# runchains.py records its own execution time, so we only track queue wait here,
# in a separate file to avoid clobbering runchains.py's runtime.txt.
# Output path comes from config.py, the single source of truth (the #SBATCH
# directives above can't read it and must stay in sync manually).
output_path="$(python3 -c 'from config import OUTPUT_PATH; print(OUTPUT_PATH)')"
run_dir="${output_path}/${SLURM_JOB_NAME}"
mkdir -p "$run_dir"

IFS='|' read -r submit_time start_time <<< "$(sacct -j "$SLURM_JOB_ID" --format=Submit,Start -n -P | head -1)"
queue_seconds=$(( $(date -d "$start_time" +%s) - $(date -d "$submit_time" +%s) ))

{
    echo "---- SLURM queue timing ----"
    echo "Submitted: $submit_time"
    echo "Started:   $start_time"
    echo "Queue wait: ${queue_seconds} seconds"
} >> "${run_dir}/queue_time.txt"

python runchains.py