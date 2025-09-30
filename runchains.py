"""
- TODO 1: Find a way to make the general model things (HOD, SHMR, mass_function, etc) apply to every likelihood
"""


import cobaya

# updated_info_minimizer, minimizer = run("/global/homes/c/cpopik/CAPPIBARAS/runchains.yaml", force=True)

info_from_yaml = cobaya.yaml.yaml_load_file("/global/homes/c/cpopik/CAPPIBARAS/runchains.yaml")

updated_info_minimizer, minimizer = cobaya.run(info_from_yaml, minimize=True, force=True)
model = minimizer.model.likelihood['tsztest']
minimum = minimizer.products()["minimum"]

# pars = [k for k, v in info_from_yaml['params'].items() if isinstance(v, dict) and "prior" in v]
# parslabel = [v['latex'] for k, v in info_from_yaml['params'].items() if isinstance(v, dict) and "prior" in v]
# pars0 = {k: v['ref'] for k, v in info_from_yaml['params'].items() if isinstance(v, dict) and "prior" in v}
# parsfit = {k: minimum[k] for k, v in info_from_yaml['params'].items() if isinstance(v, dict) and "prior" in v}

