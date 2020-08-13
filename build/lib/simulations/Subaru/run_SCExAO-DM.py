"""
run_SCExAO-DM
KD
Feb 2020

This is the starting point to run the SCExAO-DM prescription. From here, you can turn on/off toggles, change AO
settings, view different planes, and make other changes to running the prescription without changing the base
prescription or the default params themselves.

"""
import numpy as np

from medis.params import sp, tp, iop, ap
from medis.CDI import cdi, cdi_postprocess
from medis.utils import dprint
import medis.optics as opx
from medis.plot_tools import view_spectra, view_timeseries, quick2D, plot_planes
import medis.medis_main as mm

#################################################################################################
#################################################################################################
#################################################################################################
testname = 'SCExAO-DM-test'
iop.update_datadir(f"/home/captainkay/mazinlab/MKIDSim/CDIsim_data/")
iop.update_testname(testname)
iop.makedir()

# Telescope
tp.prescription = 'SCExAO_DM'
tp.entrance_d = 0.051  # diameter of optics in SCExAO train are 2 inches=0.051 m

# Simulation & Timing
sp.numframes = 7
sp.closed_loop = False

# Grid Parameters
sp.focused_sys = False
sp.beam_ratio = 0.08  # parameter dealing with the sampling of the beam in the pupil/focal plane
sp.grid_size = 512  # creates a nxn array of samples of the wavefront
sp.maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts) # set to grid_size if undesired

# Wavelength
ap.n_wvl_init = 3  # initial number of wavelength bins in spectral cube (later sampled by MKID detector)
ap.n_wvl_final = None  # final number of wavelength bins in spectral cube after interpolation (None sets equal to n_wvl_init)
ap.interp_wvl = False  # Set to interpolate wavelengths from ap.n_wvl_init to ap.n_wvl_final
ap.wvl_range = np.array([800, 1400]) / 1e9  # wavelength range in [m] (formerly ap.band)

# CDI
cdi.use_cdi = True
cdi.probe_w = 10  # [actuator coordinates] width of the probe
cdi.probe_h = 30  # [actuator coordinates] height of the probe
cdi.probe_center = (10,10)  # [actuator coordinates] center position of the probe
cdi.probe_amp = 2e-8  # [m] probe amplitude, scale should be in units of actuator height limits
cdi.which_DM = 'tweeter'
cdi.phs_intervals = np.pi/3
cdi.phase_integration_time = 0.01

# Toggles for Aberrations and Control
tp.act_tweeter = 49
tp.use_aber = False
tp.add_zern = False  # Just a note: zernike aberrations generate randomly each time the telescope is run, so introduces
                     # potentially inconsistent results
tp.use_ao = True
sp.skip_functions = []  # skip_functions is based on function name, alternate way of on/off than the toggling
                    # 'coronagraph' 'deformable_mirror' 'add_aber'
# Plotting
sp.show_wframe = True  # plot white light image frame
sp.show_spectra = False  # Plot spectral cube at single timestep
sp.spectra_cols = 3  # number of subplots per row in view_spectra
sp.show_tseries = True# Plot full timeseries of white light frames
sp.tseries_cols = 3  # number of subplots per row in view_timeseries
sp.show_planes = True
sp.maskd_size = 256
sp.verbose = False

# Saving
sp.save_to_disk = False  # save obs_sequence (timestep, wavelength, x, y)
sp.save_list = ['tweeter',   'detector']  # list of locations in optics train to save 'entrance_pupil',
                # 'entrance_pupil','post-DM-focus', 'coronagraph',

if __name__ == '__main__':
    # =======================================================================
    # Run it!!!!!!!!!!!!!!!!!
    # =======================================================================
    sim = mm.RunMedis(name=testname, product='fields')
    observation = sim()
    cpx_sequence = observation['fields']
    sampling = observation['sampling']

    # =======================================================================
    # Focal Plane Processing
    # =======================================================================
    # cpx_sequence = (n_timesteps ,n_planes, n_waves_init, n_astro_bodies, nx ,ny)
    cpx_sequence = opx.interp_wavelength(cpx_sequence, 2)  # interpolate over wavelength
    focal_plane = opx.extract_plane(cpx_sequence, 'detector')  # eliminates astro_body axis
    # convert to intensity THEN sum over object, keeping the dimension of tstep even if it's one
    focal_plane = np.sum(opx.cpx_to_intensity(focal_plane), axis=2)
    fp_sampling = np.copy(sampling[cpx_sequence.shape[1]-1,:])  # numpy arrays have some weird effects that make copying the array necessary

    # Also img_tseries
    img_tseries = np.sum(focal_plane, axis=1)  # sum over wavelength

    # ======================================================================
    # CDI
    # ======================================================================
    # if cdi.use_cdi:
    #     cdi_postprocess(cpx_sequence, fp_sampling, plot=True)
    #     cdi.save_tseries(img_tseries)
    #     cdi.save_cout_to_disk()

    # =======================================================================
    # Plotting
    # =======================================================================
    # White Light, Last Timestep
    if sp.show_wframe:
        # vlim = (np.min(spectralcube) * 10, np.max(spectralcube))  # setting z-axis limits
        img = np.sum(focal_plane[sp.numframes-1], axis=0)  # sum over wavelength
        quick2D(opx.extract_center(img), #focal_plane[sp.numframes-1]),
                title=f"White light image at timestep {sp.numframes} \n"  # img
                           f"AO={tp.use_ao}, CDI={cdi.use_cdi} ",
                           # f"Grid Size = {sp.grid_size}, Beam Ratio = {sp.beam_ratio} ",
                           # f"sampling = {sampling*1e6:.4f} (um/gridpt)",
                logZ=True,
                dx=fp_sampling[0],
                vlim=(None,None))  # (1e-3, 1e-1)

    # Plotting Spectra at last tstep
    if sp.show_spectra:
        tstep = sp.numframes-1
        view_spectra(focal_plane[sp.numframes-1],
                      title=f"Intensity per Spectral Bin at Timestep {tstep} \n"
                            f" AO={tp.use_ao}, CDI={cdi.use_cdi}",
                            # f"Beam Ratio = {sp.beam_ratio:.4f}",#  sampling = {sampling*1e6:.4f} [um/gridpt]",
                      logZ=True,
                      subplt_cols=sp.spectra_cols,
                      vlim=(1e-7, 1e-3),
                      dx=fp_sampling)

    # Plotting Timeseries in White Light
    if sp.show_tseries:
        view_timeseries(img_tseries, title=f"White Light Timeseries\n"
                                            f"AO={tp.use_ao}. CDI={cdi.use_cdi}",
                        subplt_cols=sp.tseries_cols,
                        logZ=True,
                        vlim=(1e-7, 1e-4),
                        dx=fp_sampling[0])

    # Plotting Selected Plane
    if sp.show_planes:
        # vlim = [(None, None), (None, None), (None, None), (1e-7,1e-3), (1e-7,1e-3), (1e-7,1e-3)]
        vlim = [(None,None), (1e-7,1e-4), (1e-8,1e-4), (None,None)]  # (1e-2,1e-1) (7e-4, 6e-4)
        logZ = [False, False, True, True, True, True]
        if sp.save_list:
            plot_planes(cpx_sequence,
                        title=f"White Light through Optical System",
                        subplt_cols=2,
                        vlim=vlim,
                        logZ=logZ,
                        first=True,
                        dx=sampling)

    test = 1
