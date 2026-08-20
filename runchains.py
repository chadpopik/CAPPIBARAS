"""
Runs a cobaya MCMC chain. Not meant to be run directly by hand -- submit it
through one of these instead:

    python submit_chain.py --job-name RG25_run2 --yaml RG25 --cluster rusty
        Preferred. Creates OUTPUT_PATH/RG25_run2/, stages a frozen copy of
        yamls/RG25.yaml there, then sbatches it. --cluster also takes
        "nersc" or "local" (a detached tmux session instead of Slurm).
        See submit_chain.py's own docstring for the rest of its flags.

    sbatch -J RG25_run2 runchains_rusty.sh RG25
        Hand-maintained rusty-only alternative. Uses yamls/RG25.yaml live
        (not staged/frozen), so editing that yaml while the job is still
        queued changes what the job runs.

    python runchains.py [job_name] [yaml_name]
        Bare local run, no Slurm. yaml_name defaults to
        runchainsRiedGuachalla_2 if omitted.
"""
import os, sys, cobaya, yaml, time
from pathlib import Path
from cobaya.mpi import is_main_process, sync_processes, set_mpi_disabled
import SOLikelihoods

from config import OUTPUT_PATH, CAPPIBARAS_PATH

# Get name of job and (optionally) which yaml file to use. Under Slurm,
# SLURM_JOB_NAME comes from sbatch's -J/--job-name, so argv[1] (if given) is
# free to be the yaml file's base name (yamls/<name>.yaml), e.g.
#   sbatch -J RG25_run2 runchains_rusty.sh RG25
# Outside Slurm (a bare `python runchains.py` or the tmux local launcher in
# submit_chain.py), there's no SLURM_JOB_NAME, so argv[1] is the job name
# instead (matching the previous behavior) and argv[2], if given, is the
# yaml base name.
if os.environ.get("SLURM_JOB_NAME"):
    job_name = os.environ["SLURM_JOB_NAME"]
    yaml_name = sys.argv[1] if len(sys.argv) > 1 else None
else:
    job_name = sys.argv[1] if len(sys.argv) > 1 else "local_run"
    yaml_name = sys.argv[2] if len(sys.argv) > 2 else None

# Run folder for this job. submit_chain.py (or the hand-maintained
# runchains_*.sh scripts) normally creates this first -- Slurm's --output/
# --error need it to exist before the job even starts -- but create it here
# too as a fallback for a bare `python runchains.py`. exist_ok=True makes
# this safe if multiple MPI ranks reach it at the same time.
run_dir = Path(OUTPUT_PATH) / job_name
run_dir = run_dir.resolve()
run_dir.mkdir(parents=True, exist_ok=True)

# submit_chain.py's --yaml flag stages a frozen copy of yamls/<name>.yaml
# here at submission time (before sbatch even runs), so a job always uses the
# yaml as it looked when submitted, even if yamls/<name>.yaml is later edited
# while the job is still queued. Prefer that staged copy when present;
# otherwise (bare `python runchains.py`, or the hand-maintained
# runchains_*.sh path) resolve yamls/<name>.yaml live.
staged_yaml = run_dir / f"{job_name}.yaml"
if staged_yaml.exists():
    yamlfile = staged_yaml
else:
    yamlfile = f"{CAPPIBARAS_PATH}/yamls/{yaml_name}.yaml" if yaml_name else f"{CAPPIBARAS_PATH}/yamls/runchainsRiedGuachalla_2.yaml"
with open(yamlfile) as f:
    yaml_info = yaml.safe_load(f)


# Describe the sampler settings for the MCMC run
yaml_info["sampler"] = {"mcmc": {
    "Rminus1_stop": 0.01,
    "max_tries": "200d",
    "output_every": "10s",
    "learn_every": 40
}}

# try import mpi4py to use for cluster submitting jobs
try:
    from mpi4py import MPI  # noqa: F401
except Exception:
    set_mpi_disabled()


# Set the output path and have to identify chains
yaml_info["output"] = str(run_dir / job_name)
yaml_info["output_options"] = {
    "columns": {"add_chain": True}
}
yaml_info["debug"] = False

# Hand the saved yaml to each likelihood via its cobaya options block.
for like in yaml_info["likelihood"].values():
    like["YAML_FILE"] = f"{run_dir}/{job_name}.yaml"


# save newly modified yaml file there. Only rank 0 writes it, and every rank
# waits at the barrier before reading it back in via SOLikelihoods.YAML_FILE,
# so parallel MPI chains don't race to write/read the same file.
if is_main_process():
    with open(f"{run_dir}/{job_name}.yaml", "w") as f:
        yaml.safe_dump(yaml_info, f, sort_keys=False)
sync_processes()




start = time.time()

# run cobaya
try:
    updated_info, sampler = cobaya.run(yaml_info,force=True)
    status = "completed"

except Exception as e:
    status = f"failed: {e}"
    raise

finally:
    runtime = time.time() - start

    if is_main_process():
        with open(run_dir / "runtime.txt", "w") as f:
            f.write(f"Status: {status}\n")
            f.write(f"Runtime: {runtime:.2f} seconds\n")
            f.write(f"Runtime hours: {runtime/3600:.3f}\n")