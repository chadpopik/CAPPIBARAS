from Basics import *
setplot(dark=True)

from cobaya.yaml import yaml_load_file
from cobaya.run import run
import getdist.plots as gdplt



info_from_yaml = yaml_load_file("/global/homes/c/cpopik/Capybara/runchains.yaml")

updated_info_minimizer, minimizer = run(info_from_yaml, force=True)

updated_info_minimizer, minimizer = run(info_from_yaml, minimize=True, force=True)

minimum = minimizer.products()["minimum"]

folder,name = os.path.split(os.path.abspath(info_from_yaml["output"]))
gdplot = gdplt.get_subplot_plotter(chain_dir=folder)
gdplot.settings.title_limit_fontsize=14

fitpars = [k for k, v in info_from_yaml['params'].items() if isinstance(v, dict) and "prior" in v]
fitparsmark = {k: minimum[k] for k, v in info_from_yaml['params'].items() if isinstance(v, dict) and "prior" in v}
fitparslab = {v['latex'] for k, v in info_from_yaml['params'].items() if isinstance(v, dict) and "prior" in v}

gdplot.triangle_plot(name, params = fitpars,legend_labels=fitparslab, markers=fitparsmark, filled=True,title_limit=1)
plt.savefig('cobaya_TEST.png',bbox_inches='tight')
plt.close()