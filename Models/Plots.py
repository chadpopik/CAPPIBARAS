import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from config import CAPPIBARAS_PATH

plotpath = f"{CAPPIBARAS_PATH}/Models/PlotChecks"

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Plot Styling
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family':'serif', 'mathtext.fontset':'dejavuserif',
    'axes.grid':True, 'grid.linestyle': ':', 'grid.alpha': 0.5,
    'xtick.direction':'in', 'xtick.minor.visible': True, 'xtick.top':True,
    'ytick.direction':'in', 'ytick.minor.visible': True, 'ytick.right':True,
})


class BasePlots:
    def __init__(self):
        self.paper = self.name = self.__class__.__name__

    def axsetup(self, ax, ax2, filename, xlabel, ylabel, xlim, ylim, xscale, yscale):
        img = plt.imread(f"{plotpath}/{self.paper}/{filename}.png")
        ax.set(ylabel=ylabel, xlabel=xlabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
        ax2.set_axis_off(); ax2.set_zorder(0); ax.patch.set_alpha(0); ax.set_zorder(1)
        ax2.imshow(img, extent=[*ax.get_xlim(), *ax.get_ylim()], aspect='auto')

    def plot(self, filename, width, height, xlabel, ylabel, xlim, ylim, xscale, yscale, nrow=1, ncol=1):
        fig, axs = plt.subplots(nrow, ncol, figsize=(width, height), layout='constrained')
        if nrow==1 and ncol==1:
            ax2 = fig.add_subplot(111, frameon=False)
            self.axsetup(axs, ax2, filename, xlabel, ylabel, xlim, ylim, xscale, yscale)
            return fig, axs
        
        for i, ax in enumerate(axs.flatten()):
            ax2 = fig.add_subplot(int(f'{nrow}{ncol}{i+1}'), frameon=False)
            self.axsetup(ax, ax2, filename[i], 
                xlabel=xlabel[i] if isinstance(xlabel, list) else xlabel, 
                ylabel=ylabel[i] if isinstance(ylabel, list) else ylabel, 
                xlim=xlim[i] if isinstance(xlim, list) else xlim,
                ylim=ylim[i] if isinstance(ylim, list) else ylim,
                xscale=xscale[i] if isinstance(xscale, list) else xscale,
                yscale=yscale[i] if isinstance(yscale, list) else yscale)
            
        self.autofontsize(fig)
        return fig, axs
    
    def autofontsize(self, fig, label_mult=2, tick_mult=1.5, title_mult=5, legend_mult=5):
        """
        Automatically set fontsize for labels, ticks, titles, and legends
        for all axes in a figure based on figure size and subplot grid.
        """
        figsize = fig.get_size_inches()
        axes = fig.get_axes()
        nrows, ncols = axes[0].get_subplotspec().get_gridspec().get_geometry()

        fig_area = figsize[0] * figsize[1]
        n_panels = nrows * ncols
        base = max(8, np.sqrt(fig_area / n_panels))

        fs_label  = int(base * label_mult)
        fs_tick   = int(base * tick_mult)
        fs_title  = int(base * title_mult)
        fs_legend = int(base * legend_mult)

        for ax in axes:
            ax.xaxis.label.set_fontsize(fs_label)
            ax.yaxis.label.set_fontsize(fs_label)
            ax.tick_params(axis='both', labelsize=fs_tick)
            ax.title.set_fontsize(fs_title)
            if ax.get_legend():
                ax.get_legend().prop.set_size(fs_legend)

        if fig._suptitle:
            fig._suptitle.set_fontsize(int(base * title_mult * 1.2))

        return {'label': fs_label, 'tick': fs_tick, 'title': fs_title, 'legend': fs_legend}
    
    

        




class Zheng2005(BasePlots):  # ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z
    def Fig1(self, width=10, height=4):
        return self.plot(filename=['Fig1c','Fig1d'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$M (M_\odot)$', ylabel=r'$\langle N \rangle$',
            xlim=(3.15e10, 1e15), ylim=(3.1e-2, 50), xscale='log', yscale='log')
        
    def Fig3(self, width=8, height=6):
        return self.plot(filename=['Fig3a','Fig3b','Fig3c','Fig3d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'$M (M_\odot)$', ylabel=r'$\langle N \rangle$',
            xlim=(1e11, 1e15), ylim=(3.1e-2, 4.9e1), xscale='log', yscale='log')

class Nagai2007(BasePlots):  # ui.adsabs.harvard.edu/abs/2007ApJ...668....1N
    def Fig1d(self, width=5.5, height=5):
        return self.plot(filename='Fig1d', width=width, height=height,
            xlabel=r'$r/r_{500}$', ylabel=r'$P(r)/P_{500}$',
            xlim=(4.5e-2, 2.3), ylim=(2.4e-3, 3.8e2), xscale='log', yscale='log')

    def Fig2d(self, width=5.5, height=5):
        return self.plot(filename='Fig2d', width=width, height=height,
            xlabel=r'$r/r_{500}$', ylabel=r'$P(r)/P_{500}$',
            xlim=(4.5e-2, 2.3), ylim=(2.4e-3, 1.9e2), xscale='log', yscale='log')

    def FigA1(self, width=5.5, height=5):
        return self.plot(filename='FigA1', width=width, height=height,
            xlabel=r'$r/r_{500}$', ylabel=r'$P(r)/P_{500}$',
            xlim=(4.5e-2, 2.3), ylim=(2.4e-3, 1.9e2), xscale='log', yscale='log')



class Arnaud2010(BasePlots):  # ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A
    def Fig8(self, width=6, height=5):
        return self.plot(filename='Fig8', width=width, height=height,
            xlabel=r'Radius $(R_{500})$', ylabel=r'$P/P_{500}$',
            xlim=(7.5e-3, 5.2), ylim=(2.5e-4, 3.4e2), xscale='log', yscale='log')
        
        

class More2015(BasePlots):  # ui.adsabs.harvard.edu/abs/2015ApJ...806....2M
    def Fig2(self, width=6, height=6):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$M\ [h^{-1} M_\odot]$', ylabel=r'$\langle N\rangle_M$',
            xlim=(1e12, 3.1e15), ylim=(1e-1, 1e1), xscale='log', yscale='log')

    def Fig4(self, width=12, height=4):
        return self.plot(filename=['Fig4a','Fig4b','Fig4c'], nrow=1, ncol=3, width=width, height=height,
            xlabel=r'$M \ [h^{-1}M_\odot]$', ylabel=r'$\langle N\rangle_M$',
            xlim=(3.2e11, 3.1e15), ylim=(1e-2, 2.8e2), xscale='log', yscale='log')



class Battaglia2011(BasePlots):  # ui.adsabs.harvard.edu/abs/2012ApJ...758...75B
    def Fig1(self, width=16, height=6):
        return self.plot(filename=['Fig1a','Fig1b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$r/R_{200}$', ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$',
            xlim=(3e-2, 3), ylim=(8e-4, 2.35e-1), xscale='log', yscale='log')
        
    def Fig2(self, width=16, height=6):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$r/R_{200}$', ylabel=r'$P_\text{th}/P_{200}(r/R_{200})^3$',
            xlim=(3e-2, 3), ylim=(8e-4, 2.35e-1), xscale='log', yscale='log')
    
    

class Battaglia2016(BasePlots):  # ui.adsabs.harvard.edu/abs/2016JCAP...08..058B
    def Fig5(self, width=14, height=6):
        return self.plot(filename=['Fig5a','Fig5b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$x=r/R_{200}$', ylabel=r'$\bar{\rho}(x)x^2/f_b\rho_\text{crit}(z)$',
            xlim=(7e-2, 4), ylim=(1e1, 1e2), xscale='log', yscale='log')



class Planck2013(BasePlots):  # ui.adsabs.harvard.edu/abs/2013A%26A...550A.131P
    def Fig4(self, width=14, height=6):
        return self.plot(filename=['Fig4a','Fig4b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$R/R_{500}$', ylabel=r'$P/P_{500}/</(M)>$',
            xlim=(0.01, 10), ylim=(1e-3, 1e2), xscale='log', yscale='log')
        
    def Fig6(self, width=6, height=5):
        return self.plot(filename='Fig6', width=width, height=height,
            xlabel=r'$R/R_{500}$', ylabel=r'$P/P_{500}/</(M)>$',
            xlim=(0.01, 10), ylim=(1e-3, 1e2), xscale='log', yscale='log')



class Vikram2017(BasePlots):  # ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V
    def Fig3(self, width=16, height=10):
        return self.plot(filename=['Fig3a','Fig3b', 'Fig3c','Fig3d','Fig3e','Fig3f'], nrow=2, ncol=3, width=width, height=height,
            xlabel=r'$r$ [Mpc]', ylabel=r'$\xi^s_{y, g}(r)$',
            xlim=(0.01, 10), ylim=[(1e-10, 6.7e-8), (1e-10, 7.7e-8), (1e-10, 1.9e-7), (5e-10, 0.95e-6), (1e-9, 3.6e-6), (1e-9, 1.9e-5)], xscale='log', yscale='log')



class Kravtsov2018(BasePlots):  # ui.adsabs.harvard.edu/abs/2018AstL...44....8K
    def Fig4(self, width=4, height=4):
        return self.plot(filename='Fig4', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$',
            xlim=(1.8e13, 2e15), ylim=(5e10, 2.5e13), xscale='log', yscale='log')

    def Fig7(self, width=6, height=6):
        return self.plot(filename='Fig7', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}(<r_{500}) \ [M_\odot]$',
            xlim=(3.2e13, 2e15), ylim=(3.1e11, 6.2e13), xscale='log', yscale='log')

    def Fig8(self, width=6, height=6):
        return self.plot(filename='Fig8', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{sat}}(<r_{500}) \ [M_\odot]$',
            xlim=(3.2e13, 2e15), ylim=(3.1e11, 6.2e13), xscale='log', yscale='log')

    def Fig9(self, width=6, height=6):
        return self.plot(filename='Fig9', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}}/(M_{*, \text{BCG}}+M_{*, \text{sat}})$',
            xlim=(2e13, 2e15), ylim=(0, 1), xscale='log')

    def Fig10(self, width=6, height=6):
        return self.plot(filename='Fig10', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}} \ [M_\odot]$',
            xlim=(1e10, 4e15), ylim=(1e8, 2e13), xscale='log', yscale='log')

    def Fig11(self, width=6, height=6):
        return self.plot(filename='Fig11', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 4e15), ylim=(1e-3, 1.55), xscale='log', yscale='log')

    def Fig12(self, width=6, height=6):
        return self.plot(filename='Fig12', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 4e15), ylim=(3.1e-3, 1), xscale='log', yscale='log')

    def Fig13(self, width=6, height=6):
        return self.plot(filename='Fig13', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{cen}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 1e15), ylim=(1e-3, 1.55), xscale='log', yscale='log')

    def Fig14(self, width=6, height=6):
        return self.plot(filename='Fig14', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_{*, \text{tot}}/M_{200}/(\Omega_b/\Omega_m)$',
            xlim=(1e10, 1e15), ylim=(3.1e-3, 1), xscale='log', yscale='log')

    def Fig15(self, width=15, height=5):
        return self.plot(filename=['Fig15a','Fig15b','Fig15c'], nrow=1, ncol=3, width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_*/M_{500}/(\Omega_b/\Omega_m)$',
            xlim=(1.3e11, 1.6e15), ylim=(3e-3, 1), xscale='log', yscale='log')

    def Fig16(self, width=6, height=6):
        return self.plot(filename='Fig16', width=width, height=height,
            xlabel=r'$M_{500} \ [M_\odot]$', ylabel=r'$M_{*, \text{BCG}} \ [M_\odot]$',
            xlim=(1.8e13, 2e15), ylim=(2.5e10, 2.5e13), xscale='log', yscale='log')

    def Fig17(self, width=6, height=6):
        return self.plot(filename='Fig17', width=width, height=height,
            xlabel=r'$M_{200} \ [M_\odot]$', ylabel=r'$M_* \ [M_\odot]$',
            xlim=(1e9, 2e15), ylim=(1e7, 5e12), xscale='log', yscale='log')
        
        
class Moser2021(BasePlots):  # ui.adsabs.harvard.edu/abs/2021ApJ...919....2M
    def Fig2(self, width=14, height=6):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$r/r_{200c}$', r'$r/r_{200c}$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
            xlim=[[8e-2,6.2e0],[8e-2,6.2e0]], ylim=[[5e-31,1.2e-26],[1e-16,1.6e-11]], xscale='log', yscale='log')

    def Fig3(self, width=14, height=6):
        return self.plot(filename=['Fig3a','Fig3b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$M_h \ [M_\odot]$', r'$\log_10(M^*)\ (M_\odot)$'], ylabel=[r'$M_s \ [M_\odot]$', ''],
            xlim=[[10.8, 16],[10.6,11.8]], ylim=[[7,12.5],[2,5.5e4]], xscale=['linear','linear'], yscale=['linear','log'])

    def Fig4row1(self, width=14, height=6):
        return self.plot(filename=['Fig4a','Fig4b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R (\text{Mpc})$', r'$R (\text{Mpc})$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
            xlim=[[7.5e-3,1.1e1],[7.5e-3,1.1e1]], ylim=[[6.5e-31,4e-26],[1.5e-16,1.1e-11]], xscale=['log','log'], yscale=['log','log'])

    def Fig6col1(self, width=14, height=6):
        return self.plot(filename=['Fig6a','Fig6c'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R (\text{Mpc})$', r'$R (\text{Mpc})$'], ylabel=[r'$\rho_\text{gas} [\text{g cm}^{-3}]$', r'$P_\text{th} [\text{g} \text{cm}^{-1} \text{s}^{-2}]$'],
            xlim=[[7.5e-3,1.1e1],[7.5e-3,1.1e1]], ylim=[[6.5e-31,4e-26],[1.5e-16,1.1e-11]], xscale=['log','log'], yscale=['log','log'])



class Schaan2021(BasePlots):  # ui.adsabs.harvard.edu/abs/2021PhRvD.103f3513S
    def Fig2(self, width=6, height=4):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$N_\text{galaxies}$',
            xlim=(0, 0.7), ylim=(0, 3.2e4),
            xscale='linear', yscale='linear')

    def Fig3(self, width=5, height=4):
        return self.plot(filename='Fig3', width=width, height=height,
            xlabel=r'$M_\text{vir} \ [M_\odot]$', ylabel=r'$N_\text{galaxies}/N_\text{total}$',
            xlim=(1e11, 1e15), ylim=(0, 6.9e-14), xscale='log', yscale='linear')

    def Fig5(self, width=6, height=4):
        return self.plot(filename='Fig5', width=width, height=height,
            xlabel=r'$\theta \ [\text{arcmin}]$', ylabel=r'$B(\theta)/B(0)$',
            xlim=(8e-2, 1.25e1), ylim=(1e-3, 2e0), xscale='log', yscale='log')

    def Fig7a(self, width=5, height=4):
        return self.plot(filename='Fig7a', width=width, height=height,
            xlabel=r'$R \ [\text{arcmin}]$', ylabel=r'$T_\text{kSZ} [\mu \text{K} \cdot \text{arcmin}^2]$',
            xlim=(0.75, 6.3), ylim=(3e-2, 7e1), xscale='linear', yscale='log')
        
    def Fig9(self, width=5, height=4):
        return self.plot(filename='Fig9', width=width, height=height,
            xlabel=r'$R \ [\text{arcmin}]$', ylabel=r'$T_\text{tSZ+dust} [\mu \text{K} \cdot \text{arcmin}^2]$',
            xlim=(0.75, 6.3), ylim=(-25, 0.3), xscale='linear', yscale='log')

    def Fig31(self, width=5, height=4):
        return self.plot(filename='Fig31', width=width, height=height,
            xlabel=r'$M_* \ [M_\odot]$', ylabel=r'$N_\text{galaxies}/N_\text{total}$',
            xlim=(2e10, 2e12), ylim=(0, 5.45e-12), xscale='log', yscale='linear')



class Amodeo2021(BasePlots):  # ui.adsabs.harvard.edu/abs/2021PhRvD.103f3514A
    def Fig6_row1(self, width=16, height=6):
        return self.plot(filename=['Fig6a','Fig6b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R [\text{Mpc}]$', r'$R [\text{Mpc}]$'],
            ylabel=[r'$\rho_\text{gas} \ [\text{g cm}^{-3}]$', r'$P_\text{th} \ [\text{erg cm}^{-3}]$'],
            xlim=[[7.7e-2, 1.25e1],[7.7e-2, 1.25e1]], ylim=[[5e-32,4.1e-26],[1.5e-16,2.5e-12]],
            xscale='log', yscale='log')

    def Fig6_row2(self, width=16, height=6):
        return self.plot(filename=['Fig6c','Fig6d'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R [\text{arcmin}]$', r'$R [\text{arcmin}]$'],
            ylabel=[r'$\rho^\text{2D}_\text{gas} \ [\text{g cm}^{-3}] \cdot \text{Mpc}$', r'$P^\text{2D}_\text{th} \ [\text{erg cm}^{-3}] \cdot \text{Mpc} $'],
            xlim=[[0.8,6.1],[0.8,6.1]], ylim=[[6.9e-5,1.5e-3],[3e10,9.5e11]],
            xscale='linear', yscale='log')

    def Fig6_row3(self, width=15, height=5):
        return self.plot(filename=['Fig6e','Fig6f'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R [\text{arcmin}]$', r'$R [\text{arcmin}]$'],
            ylabel=[r'$T_\text{kSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$'],
            xlim=[[0.75,6.25],[0.75,6.25]], ylim=[[1.2e-1,4e1],[-22,1.25]],
            xscale=['linear','linear'], yscale=['log','linear'])

    def Fig7(self, width=16, height=6):
        return self.plot(filename=['Fig7a','Fig7b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$r/R_\text{200c}$', r'$r/R_\text{200c}$'],
            ylabel=[r'$\rho_\text{gas}(r) / \rho_c$', r'$P_\text{th} (r) / / P_{200}$'],
            xlim=[[7.3e-3,3.2e1],[7.3e-3,3.2e1]], ylim=[[2.7e-4,8.2e3],[5.5e-8,8.2e1]],
            xscale='log', yscale='log')

    def Fig11_row2(self, width=20, height=5):
        return self.plot(filename=['Fig11b','Fig11c','Fig11d'], nrow=1, ncol=3, width=width, height=height,
            xlabel=[r'$R [\text{arcmin}]$',r'$R [\text{arcmin}]$',r'$R [\text{arcmin}]$'],
            ylabel=[r'$I \ [\text{kJy/sr}]$',r'$I \ [\text{kJy/sr}]$',r'$I \ [\text{kJy/sr}]$'],
            xlim=[[1.5,6.2],[1.5,6.2],[1.5,6.2]], ylim=[[-0.2,1.6],[-0.3,2.8],[-0.3,2.7]],
            xscale='linear', yscale='linear')

    def Fig11_row3(self, width=15, height=5):
        return self.plot(filename=['Fig11e','Fig11f'], nrow=1, ncol=2, width=width, height=height,
            xlabel=[r'$R [\text{arcmin}]$', r'$R [\text{arcmin}]$'],
            ylabel=[r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$', r'$T_\text{tSZ} \ [\mu\text{K} \cdot \text{arcmin}^2]$'],
            xlim=[[1.5,6.2],[1.5,6.2]], ylim=[[-22.5,3.5],[-27.5,2.5]],
            xscale='linear', yscale='linear')



class Linke2022(BasePlots):  # ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L
    def Fig2(self, width=8, height=6):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'Halo mass $m \ [M_\odot]$', ylabel=r'$\langle N^a|m\rangle$',
            xlim=(1e11, 1e15), ylim=(1e-4, 1e2), xscale='log', yscale='log')

    def Fig8(self, width=10, height=4):
        return self.plot(filename=['Fig8a','Fig8b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'Halo mass $m \ [M_\odot]$', ylabel=r'HOD $\langle N^a|m\rangle$',
            xlim=(1e11, 1e15), ylim=(1e-4, 1e2), xscale='log', yscale='log')



class Kusiak2022(BasePlots):  # ui.adsabs.harvard.edu/abs/2022PhRvD.106l3517K
    def Fig2(self, width=6, height=4):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$\frac{1}{N_g^\text{tot}} \frac{dN_g}{dz}$',
            xlim=(0, 4), ylim=(-0.06, 1.3), xscale='linear', yscale='linear')

    def Fig8_col1(self, width=6, height=10):
        return self.plot(filename=['Fig8a','Fig8c','Fig8e'], nrow=3, ncol=1, width=width, height=height,
            xlabel=r'$\ell$', ylabel=r'$10^5 \times C^{gg}_\ell$', xlim=(1e2, 1e3), ylim=(-0.01, 0.75), yscale='linear')

    def Fig9(self, width=15, height=12):
        return self.plot(filename=['Fig9a','Fig9b','Fig9c','Fig9d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'Mass [$M_\odot/h$]', ylabel=r'mean number of galaxies',
            xlim=(2e11,5e15), ylim=(1e-2, 1e2), xscale='log', yscale='log')

    def Fig15(self, width=12, height=4):
        return self.plot(filename=['Fig15a','Fig15b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$\ell$', ylabel=r'$u_\ell^\text{m}$', xlim=(7e1, 1.5e5), ylim=(-0.05, 1.05), xscale='log')



class Xu2023(BasePlots):  # ui.adsabs.harvard.edu/abs/2023ApJ...944..200X
    def Fig9(self, width=8, height=6):
        return self.plot(filename='Fig9', width=width, height=height,
            xlabel=r'$M_h \ [M_\odot]$', ylabel=r'$M_* / M_h$',
            xlim=(1e10, 1e15), ylim=(1e-4, 1e-1), xscale='log', yscale='log')

    def Fig7(self, width=6, height=6):
        return self.plot(filename='Fig7', width=width, height=height,
            xlabel=r'$M_h \ [h^{-1} \ M_\odot]$', ylabel=r'$M_* \ [M_\odot]$',
            xlim=(1e10, 1e15), ylim=(1e6, 1e12), xscale='log', yscale='log')



class Gao2023(BasePlots):  # ui.adsabs.harvard.edu/abs/2023ApJ...954..207G
    def Fig7a(self, width=8, height=6):
        return self.plot(filename='Fig7a', width=width, height=height,
            xlabel=r'$\log(M_h) \ [M_\odot /h]$', ylabel=r'$\log(M_*) \ [M_\odot]$',
            xlim=(10, 15), ylim=(7.5, 12), xscale='linear', yscale='linear')

    def Fig2(self, width=10, height=3.5):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$\log M_* \ [M_\odot]$', ylabel=r'$n \ [h^3 \text{Mpc}^{-3} \text{dex}^{-1}]$',
            xlim=[(9, 12.5), (7, 12.5)], ylim=(5e-7, 2e-3), xscale='linear', yscale='log')



class Yuan2023(BasePlots):  # ui.adsabs.harvard.edu/abs/2024MNRAS.530..947Y
    def Fig1(self, width=6, height=3.5):
        return self.plot(filename='Fig1', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$n(z) \ [h^3 \text{Mpc}^{-3}]$',
            xlim=(0.4, 2.3), ylim=(3e-6, 2e-3), yscale='log')

    def Fig7(self, width=14, height=6):
        return self.plot(filename=['Fig7a','Fig7b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$M_h [h^{-1} M_\odot]$', ylabel=r'$N_\text{gal}$',
            xlim=(1e12, 1e15), ylim=(1e-2, 1e1), xscale='log', yscale='log')

    def Fig10(self, width=6, height=5):
        return self.plot(filename='Fig10', width=width, height=height,
            xlabel=r'$M_h \ [h^{-1} M_\odot]$', ylabel=r'$N^{(c+s)}_\text{gal}$',
            xlim=(1e12, 1e15), ylim=(3e-2, 1e1), xscale='log', yscale='log')

    def Fig14(self, width=6, height=5):
        return self.plot(filename='Fig14', width=width, height=height,
            xlabel=r'$M_h \ [h^{-1} M_\odot]$', ylabel=r'$N_\text{gal}$',
            xlim=(1e12, 1e15), ylim=(3e-2, 1e1), xscale='log', yscale='log')



class Kou2023(BasePlots):  # ui.adsabs.harvard.edu/abs/2023A%26A...675A.149K
    def Fig4(self, width=9, height=6):
        return self.plot(filename='Fig4', width=width, height=height,
            xlabel=r'$\ell$', ylabel=r'$\ell(\ell+1)C_\ell^{gg}/(2\pi)$',
            xlim=(3.9e1, 4.1e3), ylim=(2e-2, 1.1e1), xscale='log', yscale='log')

    def Fig6(self, width=12, height=8):
        return self.plot(filename=['Fig6a','Fig6b','Fig6c','Fig6d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'$\ell$', ylabel=r'$\ell(\ell+1)C_\ell^{gy}/(2\pi)$',
            xlim=(4.1e1, 1.3e3), ylim=(2.5e-9, 2.5e-7), xscale='log', yscale='log')

    def Fig8a(self, width=9, height=6):
        return self.plot(filename='Fig8a', width=width, height=height,
            xlabel=r'$M \ [M_\odot]$', ylabel=r'$\langle N | M \rangle$',
            xlim=(1e12, 2e15), ylim=(5e-4, 3e1), xscale='log', yscale='log')



class RiedGuachalla2025(BasePlots):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112j3512R
    def Fig2(self, width=6, height=5):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$z$', ylabel=r'Number',
            xlim=(0.365, 1.135), ylim=(0, 35e3), xscale='linear', yscale='linear')

    def Fig3(self, width=6, height=5):
        return self.plot(filename='Fig3', width=width, height=height,
            xlabel=r'Stellar Mass $M_* \ [M_\odot]$', ylabel=r'Number',
            xlim=(2.6e10, 3.5e12), ylim=(5.5e-1, 1.5e5), xscale='log', yscale='log')
        
    def Fig8(self, width=10, height=3.5):
        return self.plot(filename=['Fig8a','Fig8b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'$R \ [\text{arcmin}]$', ylabel=r'$T_\text{kSZ} \ [\mu \text{K arcmin}^2]$',
            xlim=(0.525, 6.525), ylim=(0.1, 20), xscale='linear', yscale='log')
        
    def Fig11(self, width=5, height=3.5):
        return self.plot(filename='Fig11', width=width, height=height,
            xlabel=r'$R \ [\text{arcmin}]$', ylabel=r'$T_\text{kSZ} \ [\mu \text{K arcmin}^2]$',
            xlim=(0.75, 6.4), ylim=(0.045, 35), xscale='linear', yscale='log')
        
    def Fig12(self, width=5, height=3.5):
        return self.plot(filename='Fig12', width=width, height=height,
            xlabel=r'$R \ [\text{arcmin}]$', ylabel=r'$T_\text{kSZ} \ [\mu \text{K arcmin}^2]$',
            xlim=(0.75, 6.4), ylim=(0.065, 28), xscale='linear', yscale='log')

    def Fig20(self, width=6, height=4.5):
        return self.plot(filename='Fig20', width=width, height=height,
            xlabel=r'$z$', ylabel=r'Number',
            xlim=(0.1, 1.15), ylim=(0, 60e3), xscale='linear', yscale='linear')



class White2022(BasePlots):  # ui.adsabs.harvard.edu/abs/2022JCAP...02..007W
    def Fig2(self, width=8, height=4):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$d \text{ln} N/dz$ (actually $\frac{1}{N}\frac{dN}{dz}$)',
            xlim=(0.2, 1.2), ylim=(-0.3, 7.05), xscale='linear', yscale='linear')
        
        
class Zhou2023(BasePlots):  # ui.adsabs.harvard.edu/abs/2023JCAP...11..097Z
    def Fig2(self, width=16, height=6):
        return self.plot(filename=['Fig2a','Fig2b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'Redshift', ylabel=r'$N (\text{deg}^{-2})$',
            xlim=(0.15, 1.25), ylim=[(0, 28), (0, 75)], xscale='linear', yscale='linear')
        
    def Fig3(self, width=16, height=6):
        return self.plot(filename=['Fig3a','Fig3b'], nrow=1, ncol=2, width=width, height=height,
            xlabel=r'Redshift', ylabel=r'$10^{3} n(z) (h^{3} \ \text{Mpc}^{-3})$',
            xlim=(0.15, 1.1), ylim=[(0, 0.65), (0, 1.65)], xscale='linear', yscale='linear')
        
        
class Liu2025(BasePlots):  # ui.adsabs.harvard.edu/abs/2025PhRvD.112h3561L
    def Fig2(self, width=6, height=5):
        return self.plot(filename='Fig2', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$dN/dz$ (actually $n(z)$)',
            xlim=(0.1, 1.3), ylim=(0, 10.25), xscale='linear', yscale='linear')
        
    def Fig3(self, width=12, height=10):
        return self.plot(filename=['Fig3a', 'Fig3b', 'Fig3c', 'Fig3d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'$R [\text{arcmin}]$', ylabel=r'Compton Y-parameter [arcmin$^2$]',
            xlim=(0.75, 6.25), ylim=[(-0.8e-6, 4.4e-6), (-0.8e-6, 4.4e-6), (-1.15e-6, 4.2e-6), (-1.15e-6, 4.2e-6)], xscale='linear', yscale='linear')
        
    def Fig4(self, width=12, height=10):
        return self.plot(filename=['Fig4a', 'Fig4b', 'Fig4c', 'Fig4d'], nrow=2, ncol=2, width=width, height=height,
            xlabel=r'$R [\text{arcmin}]$', ylabel=r'Compton Y-parameter [arcmin$^2$]',
            xlim=(0.75, 6.25), ylim=[(-0.45e-6, 5.5e-6), (-0.45e-6, 5.5e-6), (-0.5e-6, 5.5e-6), (-0.5e-6, 5.5e-6)], xscale='linear', yscale='linear')
    
    
class Hadzhiyska2026(BasePlots):  # ui.adsabs.harvard.edu/abs/2025arXiv251014135H
    def Fig2a(self, width=5, height=4):
        return self.plot(filename='Fig2a', width=width, height=height,
            xlabel=r'$\log (M/M_\odot)$', ylabel=r'$N_\text{gal}$',
            xlim=(9.9, 12.15), ylim=(-0.6e3, 12.7e3), xscale='linear', yscale='linear')
        
    def Fig2b(self, width=5, height=4):
        return self.plot(filename='Fig2b', width=width, height=height,
            xlabel=r'$z$', ylabel=r'$N_\text{gal}$',
            xlim=(0.06, 0.6), ylim=(-0.5e3, 11.25e3), xscale='linear', yscale='linear')
        