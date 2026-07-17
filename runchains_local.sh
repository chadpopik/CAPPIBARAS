#!/bin/bash
# Run runchains.py in a detached tmux session so it keeps going if the
# SSH/VSCode connection drops. Mirrors run_chain.sh's env setup, minus SLURM.
#
# Usage:   ./run_local.sh [job_name]
# Attach:  tmux attach -t <job_name>
# Detach:  Ctrl+b then d  (job keeps running)
# List:    tmux ls

set -euo pipefail

job_name="${1:-local_run}"

if tmux has-session -t "$job_name" 2>/dev/null; then
    echo "tmux session '$job_name' already exists. Attach with: tmux attach -t $job_name"
    exit 1
fi

# Mirror run_chain.sh's #SBATCH --output/--error naming (%x_%j), using a
# timestamp in place of the SLURM job ID since there isn't one locally.
# Output path comes from config.py, the single source of truth (the #SBATCH
# directives in runchains_NERSC.sh can't read it and must stay in sync manually).
output_path="$(PYTHONPATH=/global/homes/c/cpopik/CAPPIBARAS python3 -c 'from config import OUTPUT_PATH; print(OUTPUT_PATH)')"
run_dir="${output_path}/${job_name}"
mkdir -p "$run_dir"
run_id="$(date +%Y%m%d_%H%M%S)"
out_file="${run_dir}/${job_name}_${run_id}.out"
err_file="${run_dir}/${job_name}_${run_id}.err"

# Write the launch commands to a script file rather than passing them inline
# to tmux, since nested quoting through tmux's command parsing is fragile.
launcher="${run_dir}/${job_name}_${run_id}_launch.sh"
cat > "$launcher" <<EOF
#!/bin/bash
module load python
conda activate soliket-test2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export COBAYA_USE_FILE_LOCKING=False
cd /global/homes/c/cpopik/CAPPIBARAS
python runchains.py $job_name > "$out_file" 2> "$err_file"
EOF
chmod +x "$launcher"

tmux new-session -d -s "$job_name" "$launcher"

echo "Started tmux session '$job_name'."
echo "Attach:  tmux attach -t $job_name"
echo "Detach:  Ctrl+b then d"
echo "List:    tmux ls"
echo "Logs:    $out_file"
echo "         $err_file"
