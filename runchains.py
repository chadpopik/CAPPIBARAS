import cobaya

# updated_info_minimizer, minimizer = cobaya.run("/global/homes/c/cpopik/CAPPIBARAS/chains/runchainsLiu.yaml", force=True)
yaml_file = "/global/homes/c/cpopik/CAPPIBARAS/chains/runchainsLiu.yaml"
yaml_info = cobaya.yaml.yaml_load_file(yaml_file)
for key in yaml_info['likelihood'].keys():
    yaml_info['likelihood'][key]['yaml_file'] = yaml_file
updated_info_minimizer, minimizer = cobaya.run(yaml_info, force=True)
