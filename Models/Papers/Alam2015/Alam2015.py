"""
The Eleventh and Twelfth Data Releases of the Sloan Digital Sky Survey: Final Data from SDSS-III


ui.adsabs.harvard.edu/abs/2015ApJS..219...12A
arxiv.org/pdf/1501.00963
"""


class Studies(BaseStudy):  # The Eleventh and Twelfth Data Releases of the Sloan Digital Sky Survey: Final Data from SDSS-III, ui.adsabs.harvard.edu/abs/2015ApJS..219...12A
    subs = {'DR': ['DR10', 'DR12']
    }
    info = {
        'area': {'DR12': 9376, 'DR10': 6373.2},
        }

    info['area'] = cycle(info['area'], lambda a: a *u.deg**2)
