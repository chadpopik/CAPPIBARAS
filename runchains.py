import os, sys, cobaya, yaml, time
from pathlib import Path
import SOLikelihoods

from config import OUTPUT_PATH

# yaml file to use values from
yamlfile = "/global/homes/c/cpopik/CAPPIBARAS/runchains_final.yaml"
with open(yamlfile) as f:
    yaml_info = yaml.safe_load(f)
    
# Describe the sampler settings for the MCMC run
yaml_info["sampler"] = {"mcmc": {
    "Rminus1_stop": 0.01,
    "max_tries": "200d",
    "output_every": "10s",
    "learn_every": 40
}}



# Get Name of job
job_name = os.environ.get("SLURM_JOB_NAME", (sys.argv[1] if len(sys.argv) > 1 else "local_run"))

# Make new folder in runs for this job
run_dir = Path(OUTPUT_PATH) / job_name
run_dir = run_dir.resolve()
run_dir.mkdir(parents=True, exist_ok=True)

# Set the output path and have to identify chains
yaml_info["output"] = str(run_dir / job_name)
yaml_info["output_options"] = {
    "columns": {"add_chain": True}
}
yaml_info["debug"] = False


# save newly modified yaml file there
with open(f"{run_dir}/{job_name}.yaml", "w") as f:
    yaml.safe_dump(yaml_info, f, sort_keys=False)

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

    with open(run_dir / "runtime.txt", "w") as f:
        f.write(f"Status: {status}\n")
        f.write(f"Runtime: {runtime:.2f} seconds\n")
        f.write(f"Runtime hours: {runtime/3600:.3f}\n")