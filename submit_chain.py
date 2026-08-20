"""
Generates the .sh script that runchains.py needs (Slurm sbatch script for
rusty/nersc, or a tmux launcher for local), stages a frozen copy of the
chosen yaml into the run directory, then submits it -- mirroring
runchains_rusty.sh / runchains_NERSC.sh / runchains_local.sh by hand.

Usage:
    python submit_chain.py                                  # uses JOB_NAME/CLUSTER/YAML_NAME below
    python submit_chain.py --job-name my_run --yaml RG25
    python submit_chain.py --job-name my_run --yaml RG25 --cluster nersc --time 04:00:00 --cpus-per-task 8
    python submit_chain.py --job-name my_run --yaml RG25 --cluster local
    python submit_chain.py --job-name my_run --yaml RG25 --dry-run      # write the .sh but don't submit
"""

import argparse
import shutil
import subprocess

from config import *

# ---------------------------------------------------------------------------
# Everything a run needs, in one place. Edit these directly to change what a
# plain `python submit_chain.py` submits; each one also has a matching --flag
# (see parse_args) for a one-off override without touching this file.
# Which cluster to use is picked here too (CLUSTER), by name into LAUNCHERS
# at the bottom of this file.
# ---------------------------------------------------------------------------
JOB_NAME = "CobayaRunRusty_LiuJoint_1"  # also SLURM_JOB_NAME / run_dir name / tmux session name
YAML_NAME = "runchainsRiedGuachalla_2"  # yamls/<name>.yaml -- staged into run_dir before submission
CLUSTER = "rusty"  # "rusty", "nersc", or "local"

NODES = 1
# 4 MPI chains, for the Gelman-Rubin R-1 convergence check cobaya's mcmc sampler uses across independent chains. Only rusty currently loads openmpi/installs mpi4py (see RustyLauncher.env_setup), so this default only actually parallelizes there; nersc still launches a single plain process (NERSCLauncher.launch_cmd), so pass --ntasks 1 there to avoid reserving 4 Slurm tasks for 1 process.
NTASKS = 4
CPUS_PER_TASK = 4
TIME = "0-03:00:00"

RUSTY_PARTITION = "gen"
RUSTY_VENV = "/mnt/home/cpopik/soliket_cappibaras_venv"

NERSC_ACCOUNT = "mp107a"
NERSC_QOS = "regular"
NERSC_CONDA_ENV = "soliket-test2"

class ClusterLauncher:
    """
    Base class for one cluster's script-generation + submission. Subclasses
    fill in build_script() (render a template into a full .sh) and submit()
    (hand that .sh to sbatch, or run it directly).

    name: registry key -- must match a --cluster choice and a LAUNCHERS entry.
    """
    name = None

    def __init__(self, job_name, yaml_name, nodes, ntasks, cpus_per_task, time):
        self.job_name = job_name
        self.yaml_name = yaml_name
        self.nodes = nodes
        self.ntasks = ntasks
        self.cpus_per_task = cpus_per_task
        self.time = time

    @property
    def run_dir(self):
        # Where runchains.py's output/logs/queue-timing/staged yaml all land:
        # OUTPUT_PATH/<job_name>, matching how runchains.py itself lays out
        # its output directory.
        return OUTPUT_PATH / self.job_name

    def stage_yaml(self):
        # Freeze yamls/<yaml_name>.yaml into run_dir/<job_name>.yaml *before*
        # the job is submitted, so a queued/running job always uses the yaml
        # as it looked at submission time -- editing yamls/<yaml_name>.yaml
        # afterwards (to prep the next run) can't change a job already in
        # flight. runchains.py looks for this file first and only falls back
        # to reading yamls/<yaml_name>.yaml live if it isn't there.
        src = CAPPIBARAS_PATH / "yamls" / f"{self.yaml_name}.yaml"
        dst = self.run_dir / f"{self.job_name}.yaml"
        shutil.copy2(src, dst)

    def build_script(self):
        raise NotImplementedError

    def submit(self, sh_path):
        raise NotImplementedError

    def write_and_submit(self, dry_run):
        # run_dir must exist before Slurm's --output/--error can be used, and
        # before the yaml can be staged into it -- both must happen ahead of
        # sbatch/tmux, not inside the job like runchains.py's own fallback
        # mkdir does for a bare `python runchains.py`.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_yaml()

        script_text = self.build_script()
        sh_path = self.run_dir / f"{self.job_name}_{self.name}.sh"
        sh_path.write_text(script_text)
        sh_path.chmod(0o755)  # executable, so it can also be run/inspected by hand later
        print(f"Wrote {sh_path}")

        if dry_run:
            print("--dry-run set, not submitting.")
            return
        self.submit(sh_path)


class SlurmLauncher(ClusterLauncher):
    """Shared by rusty/nersc: both render template and submit via sbatch."""

    # `{resource_directives}`, `{env_setup}`, `{launch_cmd}`, `{out_path}`,
    # and `{err_path}` are filled in per-cluster by subclasses; every other
    # placeholder comes from build_script() below. Any literal `{`/`}` that
    # must survive into the bash output (the `${...}` variable expansions and
    # the `{ ... }` grouping block) is doubled to `{{`/`}}` so str.format()
    # doesn't try to interpret it as a field.
    template = """#!/bin/bash
#SBATCH --job-name={job_name}

#SBATCH --output={out_path}
#SBATCH --error={err_path}

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
# generation time (see ClusterLauncher.write_and_submit), so the #SBATCH
# --output/--error paths above and this run_dir are guaranteed to agree
# without any manual syncing.
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

    def resource_directives(self):
        raise NotImplementedError

    def env_setup(self):
        raise NotImplementedError

    def launch_cmd(self):
        raise NotImplementedError

    def out_path(self):
        # Fixed names rather than the usual %x_%j (job name + job id), so
        # each run_dir always has one obvious output.out/error.err instead of
        # a new pair per job id -- resubmitting under the same job name
        # overwrites the previous run's logs rather than accumulating them.
        # RustyLauncher overrides this to the literal %x-rooted directory
        # form instead (matching runchains_rusty.sh), which is equivalent in
        # practice -- %x expands to this same job_name -- but keeps the
        # directive human-readable/copy-pasteable without a job name baked in.
        return f"{self.run_dir}/output.out"

    def err_path(self):
        return f"{self.run_dir}/error.err"

    def build_script(self):
        return self.template.format(
            job_name=self.job_name,
            run_dir=self.run_dir,
            cappibaras_path=CAPPIBARAS_PATH,
            nodes=self.nodes,
            ntasks=self.ntasks,
            cpus_per_task=self.cpus_per_task,
            time=self.time,
            resource_directives=self.resource_directives(),
            env_setup=self.env_setup(),
            launch_cmd=self.launch_cmd(),
            out_path=self.out_path(),
            err_path=self.err_path(),
        )

    def submit(self, sh_path):
        subprocess.run(["sbatch", str(sh_path)], check=True)


class RustyLauncher(SlurmLauncher):
    # Rusty: just a partition, no account. openmpi is loaded so mpi4py
    # (installed into the venv) is available, letting cobaya's mcmc sampler
    # run one chain per srun task and check convergence across them via
    # Gelman-Rubin R-1 instead of a single-chain fallback.
    name = "rusty"

    def __init__(self, *, partition, venv, **kwargs):
        super().__init__(**kwargs)
        self.partition = partition
        self.venv = venv

    def resource_directives(self):
        return f"#SBATCH --partition={self.partition}"

    def env_setup(self):
        return f"module load openmpi/4.1.8\nmodule load python/3.11.11\n\nsource {self.venv}/bin/activate"

    def launch_cmd(self):
        return "srun python runchains.py"

    def out_path(self):
        return f"{OUTPUT_PATH}/%x/output.out"

    def err_path(self):
        return f"{OUTPUT_PATH}/%x/error.err"


class NERSCLauncher(SlurmLauncher):
    # NERSC (Perlmutter): CPU constraint + QOS instead of a partition, conda
    # instead of venv. mpi4py isn't set up in the conda env, so this still
    # launches a single plain process; pass --ntasks 1 here or the job
    # reserves 4 Slurm tasks for 1 process actually doing anything.
    name = "nersc"

    def __init__(self, *, account, qos, conda_env, **kwargs):
        super().__init__(**kwargs)
        self.account = account
        self.qos = qos
        self.conda_env = conda_env

    def resource_directives(self):
        return f"#SBATCH --constraint=cpu\n#SBATCH --account={self.account}\n#SBATCH -q {self.qos}"

    def env_setup(self):
        return f"module load python\n\nconda activate {self.conda_env}"

    def launch_cmd(self):
        return "python runchains.py"


class LocalLauncher(ClusterLauncher):
    # No Slurm involved: launches runchains.py in a detached tmux session
    # instead, so a run keeps going after the SSH/VSCode connection that
    # started it drops.
    name = "local"

    # The inner heredoc (delimited by LAUNCHEOF) is written out to its own
    # file and handed to tmux, rather than passed as an inline command
    # string, because nested quoting through tmux's own command parsing is
    # fragile.
    template = """#!/bin/bash
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
# ClusterLauncher.write_and_submit already created it before writing/launching
# this script, so nothing here needs to mkdir it.
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

    def __init__(self, *, venv, **kwargs):
        super().__init__(**kwargs)
        self.venv = venv

    def build_script(self):
        return self.template.format(
            job_name=self.job_name,
            run_dir=self.run_dir,
            venv=self.venv,
            cappibaras_path=CAPPIBARAS_PATH,
            cpus_per_task=self.cpus_per_task,
        )

    def submit(self, sh_path):
        # local has no scheduler to hand off to, so just run the launcher
        # script directly; it backgrounds itself via tmux.
        subprocess.run(["bash", str(sh_path)], check=True)


LAUNCHERS = {"rusty": RustyLauncher, "nersc": NERSCLauncher, "local": LocalLauncher}


def make_launcher(args):
    """Build the ClusterLauncher for args.cluster from the parsed CLI args."""
    common = dict(
        job_name=args.job_name,
        yaml_name=args.yaml,
        nodes=args.nodes,
        ntasks=args.ntasks,
        cpus_per_task=args.cpus_per_task,
        time=args.time,
    )
    if args.cluster == "rusty":
        return RustyLauncher(partition=args.partition, venv=args.venv, **common)
    if args.cluster == "nersc":
        return NERSCLauncher(account=args.account, qos=args.qos, conda_env=args.conda_env, **common)
    return LocalLauncher(venv=args.venv, **common)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Names the job, the run's output directory, and (for local) the tmux session.
    # Defaults to JOB_NAME/CLUSTER/YAML_NAME above, so these are only needed for a one-off override.
    p.add_argument("--job-name", default=JOB_NAME, help="Job name (also SLURM_JOB_NAME / run directory / tmux session name)")
    p.add_argument("--yaml", default=YAML_NAME, help="yamls/<name>.yaml to stage into the run directory and use for this run")
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
    launcher = make_launcher(args)
    launcher.write_and_submit(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
