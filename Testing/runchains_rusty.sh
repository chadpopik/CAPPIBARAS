#!/bin/bash
# Usage: sbatch -J <job_name> runchains_rusty.sh [yaml_base_name]
#   e.g. sbatch -J RG25_run2 runchains_rusty.sh RG25
#   uses yamls/RG25.yaml and names the job/run_dir RG25_run2.
#   -J overrides the --job-name default below; the yaml arg is optional and
#   falls back to runchains.py's own default when omitted.
#SBATCH --job-name=RG25_all_fix_gamma_alpha_beta

#SBATCH --output=/mnt/ceph/users/cpopik/CAPPIBARAS_runs/%x/%x_%j.out
#SBATCH --error=/mnt/ceph/users/cpopik/CAPPIBARAS_runs/%x/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=gen

module load openmpi/4.1.8
module load python/3.11.11

source /mnt/home/cpopik/soliket_cappibaras_venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export COBAYA_USE_FILE_LOCKING=False

cd /mnt/home/cpopik/CAPPIBARAS

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

srun python runchains.py "$1"
