"""
Generates the .sh script that runchains.py needs (Slurm sbatch script for
rusty/nersc, or a tmux launcher for local), then submits it, mirroring
runchains_rusty.sh / runchains_NERSC.sh / runchains_local.sh by hand.

Usage:
    python submit_chain.py                                  # uses JOB_NAME/CLUSTER/etc. below
    python submit_chain.py --job-name my_run
    python submit_chain.py --job-name my_run --cluster nersc --time 04:00:00 --cpus-per-task 8
    python submit_chain.py --job-name my_run --cluster local
    python submit_chain.py --job-name my_run --dry-run      # write the .sh but don't submit
"""

import argparse
import subprocess
from pathlib import Path

from config import *

# ---------------------------------------------------------------------------
# Everything a run needs, in one place. Edit these directly to change what a
# plain `python submit_chain.py` submits; each one also has a matching --flag
# (see parse_args) for a one-off override without touching this file.
# ---------------------------------------------------------------------------
JOB_NAME = "CobayaRunRusty_LiuJoint_1"  # also SLURM_JOB_NAME / run_dir name / tmux session name
CLUSTER = "rusty"  # "rusty", "nersc", or "local"

NODES = 1
# 4 MPI chains, for the Gelman-Rubin R-1 convergence check cobaya's mcmc sampler uses across independent chains. Only the rusty branch currently loads openmpi/installs mpi4py (see env_setup in build_script), so this default only actually parallelizes there; nersc still launches a single plain process (launch_cmd below), so pass --ntasks 1 there to avoid reserving 4 Slurm tasks for 1 process.
NTASKS = 4
CPUS_PER_TASK = 4
TIME = "1-00:00:00"

RUSTY_PARTITION = "gen"
RUSTY_VENV = "/mnt/home/cpopik/soliket_cappibaras_venv"

NERSC_ACCOUNT = "mp107a"
NERSC_QOS = "regular"
NERSC_CONDA_ENV = "soliket-test2"

# Shared by rusty and nersc, which differ in their Slurm resource directives
# (plain partition vs constraint/account/qos), env setup (venv + openmpi vs
# conda), and launch command (srun, for rusty's MPI chains, vs plain python).
# `{resource_directives}`, `{env_setup}`, and `{launch_cmd}` are filled in
# per-cluster in build_script(); every other placeholder comes from the
# `common` dict there.
# Any literal `{`/`}` that must survive into the bash output (the `${...}`
# variable expansions and the `{ ... }` grouping block) is doubled to `{{`/`}}`
# so str.format() doesn't try to interpret it as a field.
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}

#SBATCH --output={run_dir}/%x_%j.out
#SBATCH --error={run_dir}/%x_%j.err

#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
{resource_directives}

{env_setup}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export COBAYA_USE_FILE_LOCKING=False

cd {cappibaras_path}

# run_dir is resolved from config.py's OUTPUT_PATH once, in Python, at script-
# generation time (see build_script), so the #SBATCH --output/--error paths
# above and this run_dir are guaranteed to agree without any manual syncing.
run_dir="{run_dir}"

# Record time spent waiting in the SLURM queue (Submit -> Start).
# runchains.py records its own execution time, so we only track queue wait here,
# in a separate file to avoid clobbering runchains.py's runtime.txt.
# sacct is queried after the job has already started (we're running inside
# it), so Submit/Start are both already populated for this job ID.
IFS='|' read -r submit_time start_time <<< "$(sacct -j "$SLURM_JOB_ID" --format=Submit,Start -n -P | head -1)"
queue_seconds=$(( $(date -d "$start_time" +%s) - $(date -d "$submit_time" +%s) ))

{{
    echo "---- SLURM queue timing ----"
    echo "Submitted: $submit_time"
    echo "Started:   $start_time"
    echo "Queue wait: ${{queue_seconds}} seconds"
}} >> "${{run_dir}}/queue_time.txt"

{launch_cmd}
"""

# No Slurm involved: launches runchains.py in a detached tmux session instead,
# so a run keeps going after the SSH/VSCode connection that started it drops.
# The inner heredoc (delimited by LAUNCHEOF) is written out to its own file
# and handed to tmux, rather than passed as an inline command string, because
# nested quoting through tmux's own command parsing is fragile.
LOCAL_TEMPLATE = """#!/bin/bash
# Run runchains.py in a detached tmux session so it keeps going if the
# SSH/VSCode connection drops.
#
# Attach:  tmux attach -t {job_name}
# Detach:  Ctrl+b then d  (job keeps running)
# List:    tmux ls

set -euo pipefail

job_name="{job_name}"

if tmux has-session -t "$job_name" 2>/dev/null; then
    echo "tmux session '$job_name' already exists. Attach with: tmux attach -t $job_name"
    exit 1
fi

# Same run_dir (under config.py's OUTPUT_PATH) as the Slurm templates use.
# submit_chain.py's main() already created it before writing/launching this
# script, so nothing here needs to mkdir it.
run_dir="{run_dir}"
# There's no Slurm job ID to key log filenames on locally, so use a timestamp
# instead (mirrors the %j in the Slurm templates' --output/--error paths).
run_id="$(date +%Y%m%d_%H%M%S)"
out_file="${{run_dir}}/${{job_name}}_${{run_id}}.out"
err_file="${{run_dir}}/${{job_name}}_${{run_id}}.err"

launcher="${{run_dir}}/${{job_name}}_${{run_id}}_launch.sh"
cat > "$launcher" <<LAUNCHEOF
#!/bin/bash
module load python/3.11.11
source {venv}/bin/activate
export OMP_NUM_THREADS={cpus_per_task}
export MKL_NUM_THREADS={cpus_per_task}
export COBAYA_USE_FILE_LOCKING=False
cd {cappibaras_path}
python runchains.py $job_name > "$out_file" 2> "$err_file"
LAUNCHEOF
chmod +x "$launcher"

tmux new-session -d -s "$job_name" "$launcher"

echo "Started tmux session '$job_name'."
echo "Attach:  tmux attach -t $job_name"
echo "Detach:  Ctrl+b then d"
echo "List:    tmux ls"
echo "Logs:    $out_file"
echo "         $err_file"
"""


def build_script(args):
    """Render the .sh contents for args.cluster from the parsed CLI args."""
    # Fields every template needs, regardless of cluster. run_dir is where
    # runchains.py's output/logs/queue-timing all land: OUTPUT_PATH/<job_name>,
    # matching how runchains.py itself lays out its output directory.
    common = dict(
        job_name=args.job_name,
        cappibaras_path=CAPPIBARAS_PATH,
        run_dir=OUTPUT_PATH / args.job_name,
        nodes=args.nodes,
        ntasks=args.ntasks,
        cpus_per_task=args.cpus_per_task,
        time=args.time,
    )
    if args.cluster == "rusty":
        # Rusty: just a partition, no account. openmpi is loaded so mpi4py
        # (installed into the venv) is available, letting cobaya's mcmc
        # sampler run one chain per srun task and check convergence across
        # them via Gelman-Rubin R-1 instead of a single-chain fallback.
        resource_directives = f"#SBATCH --partition={args.partition}"
        env_setup = f"module load openmpi/4.1.8\nmodule load python/3.11.11\n\nsource {args.venv}/bin/activate"
        return SLURM_TEMPLATE.format(**common, resource_directives=resource_directives, env_setup=env_setup, launch_cmd="srun python runchains.py")
    if args.cluster == "nersc":
        # NERSC (Perlmutter): CPU constraint + QOS instead of a partition, conda instead of venv.
        # mpi4py isn't set up in NERSC_CONDA_ENV, so this still launches a single
        # plain process; pass --ntasks 1 here or the job reserves 4 Slurm tasks
        # for 1 process actually doing anything.
        resource_directives = f"#SBATCH --constraint=cpu\n#SBATCH --account={args.account}\n#SBATCH -q {args.qos}"
        env_setup = f"module load python\n\nconda activate {args.conda_env}"
        return SLURM_TEMPLATE.format(**common, resource_directives=resource_directives, env_setup=env_setup, launch_cmd="python runchains.py")
    # local: no Slurm fields needed beyond what's already in `common`.
    return LOCAL_TEMPLATE.format(**common, venv=args.venv)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Names the job, the run's output directory, and (for local) the tmux session.
    # Defaults to JOB_NAME/CLUSTER above, so this is only needed for a one-off override.
    p.add_argument("--job-name", default=JOB_NAME, help="Job name (also SLURM_JOB_NAME / run directory / tmux session name)")
    p.add_argument("--cluster", choices=["rusty", "nersc", "local"], default=CLUSTER)
    # Slurm resource knobs, shared across rusty/nersc; meaningless for local since
    # there's no scheduler there (cpus_per_task still sets OMP/MKL threads locally).
    # Defaults come from the constants at the top of this file.
    p.add_argument("--time", default=TIME, help="Slurm --time (ignored for local)")
    p.add_argument("--nodes", type=int, default=NODES, help="Slurm --nodes (ignored for local)")
    p.add_argument("--ntasks", type=int, default=NTASKS, help="Slurm --ntasks (ignored for local)")
    p.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK, help="Slurm --cpus-per-task, also OMP/MKL thread count")
    # Cluster-specific scheduler settings.
    p.add_argument("--partition", default=RUSTY_PARTITION, help="Slurm --partition (rusty only)")
    p.add_argument("--account", default=None, help=f"Slurm --account (nersc only; defaults to '{NERSC_ACCOUNT}'; rusty doesn't set one)")
    p.add_argument("--qos", default=NERSC_QOS, help="Slurm -q (nersc only)")
    # Environment activation, one flavor per cluster.
    p.add_argument("--venv", default=RUSTY_VENV, help="Python venv to activate (rusty/local)")
    p.add_argument("--conda-env", default=NERSC_CONDA_ENV, help="Conda environment to activate (nersc only)")
    p.add_argument("--dry-run", action="store_true", help="Write the .sh file but do not submit it")
    args = p.parse_args()
    # Only nersc needs an account; rusty's resource_directives never
    # references args.account. Resolved here (after argparse, so we know
    # which cluster was picked) rather than in add_argument.
    if args.account is None and args.cluster == "nersc":
        args.account = NERSC_ACCOUNT
    return args


def main():
    args = parse_args()
    script_text = build_script(args)

    # Write the generated script into the same run_dir (OUTPUT_PATH/<job_name>)
    # that the script's own --output/--error/queue_time.txt use, so everything
    # for a given run lives in one place under OUTPUT_PATH instead of scattering
    # a second copy of the folder structure back into the CAPPIBARAS checkout.
    # This also satisfies Slurm's requirement that --output/--error's directory
    # already exist at submission time (it can't create missing parent dirs).
    # None of the generated .sh templates mkdir it themselves. runchains.py
    # has its own fallback mkdir too, but only for a bare `python runchains.py`
    # run that bypasses this script entirely.
    run_dir = OUTPUT_PATH / args.job_name
    run_dir.mkdir(parents=True, exist_ok=True)
    sh_path = run_dir / f"{args.job_name}_{args.cluster}.sh"
    sh_path.write_text(script_text)
    sh_path.chmod(0o755)  # executable, so it can also be run/inspected by hand later
    print(f"Wrote {sh_path}")

    if args.dry_run:
        print("--dry-run set, not submitting.")
        return

    # local has no scheduler to hand off to, so just run the launcher script
    # directly; it backgrounds itself via tmux. rusty/nersc go through sbatch.
    if args.cluster == "local":
        subprocess.run(["bash", str(sh_path)], check=True)
    else:
        subprocess.run(["sbatch", str(sh_path)], check=True)


if __name__ == "__main__":
    main()
