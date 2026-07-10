import os, sys, cobaya, yaml, time
from pathlib import Path

# Get Name of job
job_name = os.environ.get("SLURM_JOB_NAME", "local_run")

# open baseline yaml file
YAML_FILE = "/global/homes/c/cpopik/CAPPIBARAS/runs/runchains_final.yaml"
with open(YAML_FILE) as f:
    yaml_info = yaml.safe_load(f)
    
# Make new folder in runs for this job
run_dir = Path("runs") / job_name
run_dir = run_dir.resolve()
run_dir.mkdir(parents=True, exist_ok=False)

# Tell the yaml file to go there
yaml_info["output"] = str(run_dir / job_name)

# save newly modified yaml file there
with open(f"{run_dir}/{job_name}.yaml", "w") as f:
    yaml.safe_dump(yaml_info, f, sort_keys=False)


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


# import sys
# import cobaya
# import SOLikelihoods
# import yaml

# # Take yaml file name from the shell script
# SOLikelihoods.YAML_FILE = sys.argv[1]

# updated_info_minimizer, minimizer = cobaya.run(SOLikelihoods.YAML_FILE, force=True)
# YAML_FILE=/global/homes/c/cpopik/CAPPIBARAS/chains/runchains_final.yaml