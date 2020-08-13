"""
run_general
RD
Jan 22 2020

This is the starting point to run the general system. This general system (formerly optics_propagate.py) is fully
customisable from this script from a series of toggles.

"""

import numpy as np

from medis.params import sp, ap, tp
from medis.utils import dprint
from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes, grid
import medis.medis_main as mm



#################################################################################################
#################################################################################################
#################################################################################################

tp.prescription = 'general_telescope'

# Companion
ap.companion = True
ap.contrast = [1e-5]
ap.companion_xy = [[15, -15]]  # units of this are in lambda/tp.entrance_d

sp.numframes = 2
sp.focused_sys = False
sp.beam_ratio = 0.3  # parameter dealing with the sampling of the beam in the pupil/focal plane
sp.grid_size = 512  # creates a nxn array of samples of the wavefront
sp.maskd_size = 512  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired
sp.closed_loop = False

# Toggles for Aberrations and Control
tp.entrance_d = 8
tp.obscure = False
tp.use_atmos = True
tp.use_ao = True
tp.ao_act = 60
tp.rotate_atmos = False
tp.rotate_sky = False
tp.f_lens = 200.0 * tp.entrance_d
tp.occult_loc = [0,0]

# Saving
sp.save_to_disk = False  # save obs_sequence (timestep, wavelength, x, y)
sp.save_list = ['detector']  # list of locations in optics train to save
# sp.skip_planes = ['coronagraph']  # ['wfs', 'deformable mirror']  # list of locations in optics train to save
sp.quick_detect = True
sp.debug = False
sp.verbose = True

if __name__ == '__main__':
    # =======================================================================
    # Run it!!!!!!!!!!!!!!!!!
    # =======================================================================

    sim = mm.RunMedis(name='general_example', product='photons')
    observation = sim()
    fp_sampling = sim.camera.platescale
    rebinned_photons = sim.camera.rebinned_cube

    dprint(f"Sampling in focal plane is {fp_sampling})
    for o in range(len(ap.contrast)+1):
        print(rebinned_photons.shape)
        datacube = rebinned_photons[o]
        print(o, datacube.shape)
        grid(datacube, logZ=True, title='Spectral Channels')
