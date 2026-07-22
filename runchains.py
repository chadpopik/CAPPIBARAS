import os, sys, cobaya, yaml, time
from pathlib import Path
from cobaya.mpi import is_main_process, sync_processes, set_mpi_disabled
import SOLikelihoods


from config import OUTPUT_PATH, CAPPIBARAS_PATH

# yaml file to use values from
yamlfile = f"{CAPPIBARAS_PATH}/yamls/runchainsLiu.yaml"
with open(yamlfile) as f:
    yaml_info = yaml.safe_load(f)
    
# Describe the sampler settings for the MCMC run
yaml_info["sampler"] = {"mcmc": {
    "Rminus1_stop": 0.01,
    "max_tries": "200d",
    "output_every": "10s",
    "learn_every": 40
}}

# mpi4py can be importable as a package but still fail at first use (e.g. its
# MPI shared library isn't on the loader path unless a matching module, like
# openmpi/4.1.8 on rusty, is loaded). cobaya.mpi only catches ImportError for
# that case, not the RuntimeError mpi4py itself raises, so probe explicitly
# here and fall back to single-process mode on any failure instead of crashing.
try:
    from mpi4py import MPI  # noqa: F401
except Exception:
    set_mpi_disabled()


# Get Name of job
job_name = os.environ.get("SLURM_JOB_NAME", (sys.argv[1] if len(sys.argv) > 1 else "local_run"))

# Run folder for this job. submit_chain.py (or the hand-maintained
# runchains_*.sh scripts) normally creates this first -- Slurm's --output/
# --error need it to exist before the job even starts -- but create it here
# too as a fallback for a bare `python runchains.py`. exist_ok=True makes
# this safe if multiple MPI ranks reach it at the same time.
run_dir = Path(OUTPUT_PATH) / job_name
run_dir = run_dir.resolve()
run_dir.mkdir(parents=True, exist_ok=True)

# Set the output path and have to identify chains
yaml_info["output"] = str(run_dir / job_name)
yaml_info["output_options"] = {
    "columns": {"add_chain": True}
}
yaml_info["debug"] = False


# save newly modified yaml file there. Only rank 0 writes it, and every rank
# waits at the barrier before reading it back in via SOLikelihoods.YAML_FILE,
# so parallel MPI chains don't race to write/read the same file.
if is_main_process():
    with open(f"{run_dir}/{job_name}.yaml", "w") as f:
        yaml.safe_dump(yaml_info, f, sort_keys=False)
sync_processes()

SOLikelihoods.YAML_FILE = f"{run_dir}/{job_name}.yaml"

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