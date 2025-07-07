from Basics import *
setplot(dark=True)

from cobaya.yaml import yaml_load_file
from cobaya.run import run
import getdist.plots as gdplt
from cobaya.model import get_model

model = get_model("/global/homes/c/cpopik/Git/Capybara/cobayatest_chains.yaml")

# info_from_yaml = yaml_load_file("/global/homes/c/cpopik/Git/Capybara/cobayatest_chains.yaml")

# updated_info_minimizer, minimizer = run(info_from_yaml, force=True)


# updated_info_minimizer, minimizer = run(info_from_yaml, minimize=True, force=True)

# minimum = minimizer.products()["minimum"]

# folder,name = os.path.split(os.path.abspath(info_from_yaml["output"]))
# gdplot = gdplt.get_subplot_plotter(chain_dir=folder)
# gdplot.settings.title_limit_fontsize=14
# gdplot.triangle_plot(name,['A0_P0','alpha_GNFW','A0_beta','A2h'],filled=True,title_limit=1,markers={'A0_P0':minimum['A0_P0'],'alphaGNFW':minimum['alpha_GNFW'],'A0_beta':minimum['A0_beta'], 'A2h': minimum['A2h']})
# plt.savefig('cobaya_TEST.png',bbox_inches='tight')
# plt.close()