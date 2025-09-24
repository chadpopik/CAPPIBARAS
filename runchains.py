"""
- TODO 1: Find a way to make the general model things (HOD, SHMR, mass_function, etc) apply to every likelihood
"""


from cobaya.run import run

updated_info_minimizer, minimizer = run("/global/homes/c/cpopik/Capybara/runchains.yaml", force=True)