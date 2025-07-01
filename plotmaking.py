import sys
sys.path.append('/global/homes/c/cpopik/')
from Basics import *


class ProfPlot():
    def __init__(self):
        fig, gs = plt.figure(figsize=(15, 5)), mpl.gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        plt.tight_layout(), fig.subplots_adjust(hspace=0, wspace=0.25)
        ax1, ax3 = plt.subplot(gs[0]), plt.subplot(gs[1])
        ax2, ax4 = plt.subplot(gs[2], sharex=ax1), plt.subplot(gs[3], sharex=ax3)
        for ax in [ax1, ax3]: ax.tick_params(labelbottom=False)
        for ax in [ax2, ax4]: 
            ax.tick_params(axis='y', labelsize=10)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter('{0:.0%}'.format))
        
        ax1.set_ylabel(r"$\overline{P}_{\text{th}}$ [dyne/cm$^2$]"), ax2.set_xlabel(r'$r$ [Mpc]')
        ax3.set_ylabel(r'$T_{tSZ}$ [$\mu$K$\cdot$arcmin$^2$]'), ax4.set_xlabel(r'$R$ [arcmin]')
        ax2.set_ylabel(r'$\Delta$', size=15), ax4.set_ylabel(r'$\Delta$', size=15)
        
        self.ax1, self.ax2, self.ax3, self.ax4 = ax1, ax2, ax3, ax4



        
        


    


    def plotmoser():
            ax1.loglog(rsSO, PthsSO, label='Moser 2021', c='k', lw=3), ax2.axhline(0, c='k')
            ax1.loglog(rsSO, PthsSO1h, c='k', lw=2, alpha=0.5, ls='--'), ax1.loglog(rsSO, PthsSO2h, c='k', alpha=0.25, ls=':')
            ax3.plot(SOprov.thta_arc, TtszSO, label='Moser 2021', c='k', lw=3), ax4.axhline(0, c='k')
            # ax3.plot(SOprov.thta_arc, TtszSO1h, c='k', marker='.', alpha=0.4, lw=0), ax3.plot(SOprov.thta_arc, TtszSO2h, c='k', alpha=0.2, ls=':')