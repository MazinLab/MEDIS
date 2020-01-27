"""
run_general
RD
Jan 22 2020

This is the starting point to run the general system. This general system (formerly optics_propagate.py) is fully
customisable from this script from a series of toggles.

"""

import numpy as np

from medis.params import iop, sp, ap, tp, cdip
from medis.utils import dprint
# import medis.optics as opx
from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes
import medis.medis_main as mm

tp.prescription = 'general_telescope'
tp.obscure = True

#################################################################################################
#################################################################################################
#################################################################################################
# Companion
ap.companion = True
ap.contrast = [1e-5]
ap.companion_xy = [[15, -15]]  # units of this still confuse me

tp.enterance_d = 8
sp.numframes = 1

sp.focused_sys = True
sp.beam_ratio = 0.14  # parameter dealing with the sampling of the beam in the pupil/focal plane
sp.grid_size = 512  # creates a nxn array of samples of the wavefront
sp.maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired

# Toggles for Aberrations and Control
tp.obscure = False
tp.use_atmos = False
tp.use_aber = False
tp.use_ao = True
tp.ao_act = 14
tp.rotate_atmos = False
tp.rotate_sky = False
tp.f_lens = 200.0 * tp.enterance_d
tp.open_ao = True
tp.include_tiptilt = False
tp.include_dm = True
tp.use_zern_ab = False
tp.use_apod = False
tp.occult_loc = [0,0]
tp.occulter_type = 'Gaussian'
tp.use_coronagraph = False

# Plotting
sp.show_wframe = True  # Plot white light image frame
sp.show_spectra = True  # Plot spectral cube at single timestep
sp.spectra_cols = 3  # number of subplots per row in view_datacube
sp.show_tseries = False  # Plot full timeseries of white light frames
sp.tseries_cols = 5  # number of subplots per row in view_timeseries
sp.show_planes = True

# Saving
sp.save_obs = False  # save obs_sequence (timestep, wavelength, x, y)
sp.save_fields = True  # toggle to turn saving uniformly on/off
sp.save_list = ['detector']  # list of locations in optics train to save

tp.aber_params = {'CPA': True,
                    'NCPA': True,
                    'QuasiStatic': False,  # or 'Static'
                    'Phase': True,
                    'Amp': False,
                    'n_surfs': 2,
                    'OOPP': [8, 4]}  # fraction of a focal length where mirror(s) is located
tp.aber_vals = {'a': [7.2e-17, 3e-17],
                  'b': [0.8, 0.2],
                  'c': [3.1, 0.5],
                  'a_amp': [0.05, 0.01]}

if __name__ == '__main__':
    # testname = input("Please enter test name: ")
    testname = 'general'
    iop.update(testname)
    iop.makedir()

    # =======================================================================
    # Run it!!!!!!!!!!!!!!!!!
    # =======================================================================
    cpx_sequence, sampling = mm.run_medis()
    dprint(cpx_sequence.shape)
    for o in range(len(ap.contrast)):
        datacube = np.sum(np.abs(cpx_sequence)**2, axis=(0,1))[:,o]
        view_spectra(datacube, logZ=True)