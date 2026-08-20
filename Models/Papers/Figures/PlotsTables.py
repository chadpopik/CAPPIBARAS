import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family':'serif', 'mathtext.fontset':'dejavuserif',
    'axes.grid':True, 'grid.linestyle': ':', 'grid.alpha': 0.5,
    'xtick.direction':'in', 'xtick.minor.visible': True, 'xtick.top':True,
    'ytick.direction':'in', 'ytick.minor.visible': True, 'ytick.right':True,
})

import sys,os
thispath = os.path.dirname(os.path.abspath(__file__))


class BasePlots2():
    def __init__(self, plotpath):
        self.plotpath = plotpath
        self.paper = self.name = self.__class__.__name__

    def axsetup(self, ax, ax2, filename, xlabel, ylabel, xlim, ylim, xscale, yscale):
        img = plt.imread(f"{self.plotpath}/{filename}.png")
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

    # Subclasses that want panel()/row()/col()/full() define `subplots` as a
    # [row][col] grid of dicts of plot() kwargs (filename, xlabel, ylabel, xlim, ylim, xscale, yscale),
    # plus a 'figsize' (width, height) used to size that panel and an optional 'name' for lookup by name.
    def _plot(self, cells, nrow, ncol, width, height):
        keys = [k for k in cells[0].keys() if k not in ('name', 'figsize')]
        if nrow == 1 and ncol == 1:
            return self.plot(width=width, height=height, **{k: cells[0][k] for k in keys})
        return self.plot(nrow=nrow, ncol=ncol, width=width, height=height,
            **{k: [c[k] for c in cells] for k in keys})

    def panel(self, row, col=None, width=None, height=None):
        if col is None:
            matches = [c for r in self.subplots for c in r if c.get('name') == row]
            if not matches:
                raise KeyError(f"no subplot named {row!r}")
            cell = matches[0]
        else:
            cell = self.subplots[row][col]
        w, h = cell['figsize']
        return self._plot([cell], 1, 1, width or w, height or h)

    def panel_on(self, ax, row, col=None):
        """Like panel(), but draws the paper's background image onto an
        existing ax (in place) instead of creating a new figure, so it can
        be dropped into a subplot grid someone else already built. The
        given ax keeps its own figure and grid position; only its axis
        limits/labels/scale and the overlaid image are set. Returns the
        (transparent, non-interactive) image axis stacked behind it."""
        if col is None:
            matches = [c for r in self.subplots for c in r if c.get('name') == row]
            if not matches:
                raise KeyError(f"no subplot named {row!r}")
            cell = matches[0]
        else:
            cell = self.subplots[row][col]
        ax2 = ax.figure.add_subplot(ax.get_subplotspec(), frameon=False)
        self.axsetup(ax, ax2, cell['filename'], cell['xlabel'], cell['ylabel'],
                     cell['xlim'], cell['ylim'], cell['xscale'], cell['yscale'])
        return ax2

    def row(self, i, width=None, height=None):
        cells = self.subplots[i]
        w = width or sum(c['figsize'][0] for c in cells)
        h = height or max(c['figsize'][1] for c in cells)
        return self._plot(cells, 1, len(cells), w, h)

    def col(self, j, width=None, height=None):
        cells = [r[j] for r in self.subplots]
        w = width or max(c['figsize'][0] for c in cells)
        h = height or sum(c['figsize'][1] for c in cells)
        return self._plot(cells, len(cells), 1, w, h)

    def full(self, width=None, height=None):
        cells = [c for r in self.subplots for c in r]
        w = width or sum(c['figsize'][0] for c in self.subplots[0])
        h = height or sum(r[0]['figsize'][1] for r in self.subplots)
        return self._plot(cells, len(self.subplots), len(self.subplots[0]), w, h)



def read_wide_table(filename):
    # csv has parameters as rows and categories (e.g. samples, mass bins) as columns,
    # with the corner cell naming that category. Transpose so categories become rows,
    # carrying the corner-cell name over to the new index (pandas otherwise drops it
    # onto columns.name instead of index.name).
    df = pd.read_csv(filename, index_col=[0])
    label = df.index.name
    df = df.T
    df.index.name = label
    return df


def splittable(df):
    # df is your original DataFrame
    val_df = df.copy()
    up_df = df.copy()
    down_df = df.copy()

    pat = re.compile(
    r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
    r'(?:\+(\d*\.?\d+(?:[eE][+-]?\d+)?))?'
    r'(?:[−-](\d*\.?\d+(?:[eE][+-]?\d+)?))?$'
    )
    pat_pm = re.compile(  # symmetric "value ± error" notation
    r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*±\s*(\d*\.?\d+(?:[eE][+-]?\d+)?)$'
    )

    for col in df.columns:  # skip the parameter-name column
        vals, ups, downs = [], [], []

        for s in df[col].map(lambda v: str(v).strip()):  # .astype(str) alone is a no-op on pandas' native str dtype, leaving NaN as a float
            s = s.replace('−', '-')  # normalize unicode minus (U+2212) to ASCII, incl. as a leading sign
            m = pat.fullmatch(s)
            m_pm = pat_pm.fullmatch(s) if not m else None
            if m:
                vals.append(float(m.group(1)))
                ups.append(float(m.group(2)) if m.group(2) else None)
                downs.append(float(m.group(3)) if m.group(3) else None)
            elif m_pm:
                vals.append(float(m_pm.group(1)))
                ups.append(float(m_pm.group(2)))
                downs.append(float(m_pm.group(2)))
            else:
                try: vals.append(float(s))
                except: vals.append(s)
                ups.append(None)
                downs.append(None)

        val_df[col] = vals
        up_df[col] = ups
        down_df[col] = downs
        
    return val_df, up_df, down_df



class ParamTable(): 
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def getcol(self, key):
        return self.df[key].values

    def getparams(self, **keys):
        df = self.df.copy()
        for k,v in keys.items():
            if isinstance(df, pd.Series):
                df = pd.DataFrame(df).T
            if k == df.index.name:
                if v in df.index: df = df.loc[v]
                else: print(f"Value {v} not in {np.unique(df.index.values)}")
            elif k not in df.columns:
                print(f"key {k} not in {df.columns.to_list()}")
                pass
            elif v in df[k].values: df = df.set_index(k).loc[v]
            else: print(f"Value {v} not in {np.unique(df[k].values)}")
        return df
    
    
    
