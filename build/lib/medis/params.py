"""
This is the main configuration file. It contains default global variables (as in they are read in by the relevant
modules) that define the parameters of the whole telescope system. These parameters can be redefined at the beginning
of the example module the user is running

Unless otherwise specified, units shall be given in:
distance: meters
time: seconds

TODO
    * Add all possible initial SP attributes here so the user knows what the possible options are. Consider adding other param types too
    * add user_params.py too so we don't have to keep changing rootdir and datadir when pushing?
"""

import numpy as np
import os
import proper
from pathlib import Path
import warnings

class IO_params:
    """
    Define file tree/structure to import and save data

    telescope.gridsize and telescope.beam_ratio affect the sampling of the simulation, so are included in filenames
    of files genereated based on the sampling parameters, and simulation_params.numframes affects how many files need
    to be generated, so are also included in filenames
    """

    def __init__(self, datadir=f'{str(Path.home())}/MKIDSim/', testname='example1'):
        # you can update the datadir and testname with iop.update_testname('your_testname') and then run iop.mkdir()
        self.datadir = datadir
        self.prescriptions_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                               'simulations')  # top level of where prescriptions found

        # Unprocessed Science Data
        self.testname = testname  # set this up in the definition line, but can update it with iop.update_testname('newname')
        self.testdir = os.path.join(self.datadir,
                                  self.testname)  # Save results in new sub-directory
        self.params_logs = os.path.join(self.testdir, 'params.pkl')

        self.fields = os.path.join(self.testdir, 'fields.h5')  # a x/y/t/w cube of data
        self.photonlist = os.path.join(self.testdir, 'photonlist.h5')  # a photon table with 4 columns
        self.camera = os.path.join(self.testdir, 'camera.pkl')  # MKIDS.Camera instance save state
        self.telescope = os.path.join(self.testdir, 'telescope.pkl')  # a telecope.Telescope instance save state

        self.atmosroot = 'atmos'
        atmosdir = "gridsz{}_bmratio{}_tsteps{}"  # fill in variable names later
        self.atmosdir = os.path.join(self.testdir, self.atmosroot, atmosdir)  # full path to FITS files

        # Aberration Metadata
        self.aberroot = 'aberrations'
        aberdir = "gridsz{}_bmratio{}_tsteps{}"
        self.aberdir = os.path.join(self.testdir, self.aberroot, aberdir)  # full path to FITS files

        self.prescopyroot = 'prescription'
        prescopydir = "{}"
        self.prescopydir = os.path.join(self.testdir, self.prescopyroot, prescopydir)  # copy of the prescription

        self.device = os.path.join(self.testdir, 'device.pkl')  # detector metadata

    def update_testname(self, new_name='example2'):
        self.__init__(datadir=self.datadir, testname=new_name)

    def update_datadir(self, new_root=f'{str(Path.home())}/MKIDSim/'):
        """Update base directory where  simulation data and output files are saved.
        Remove old directory tree first if empty"""
        self.__init__(datadir=new_root, testname=self.testname)

    def makedir(self):
        #print(self.datadir, self.testdir,  self.scidir)
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir, exist_ok=True)
        if not os.path.isdir(self.testdir):
            os.makedirs(self.testdir, exist_ok=True)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Simulation_params:
    """
    Default parameters for outputs of the simulation. What plots you want to see etc

    """
    def __init__(self):
        self.timing = True  # True will print timing statements in run_medis()
        self.num_processes = 1  # multiprocessing.cpu_count()

        # Grid Sizing/Sampling Params
        self.beam_ratio = 0.5  # parameter dealing with the sampling of the beam in the pupil/focal
                                # plane vs grid size. See Proper manual pgs 36-37 and 71-74 for discussion
        self.grid_size = 512  # creates a nxn array of samples of the wavefront
                              # must be bigger than the beam size to avoid FT effects at edges; must be factor of 2
                              # NOT the size of your detector/# of pixels
        self.maskd_size = 256  # will truncate grid_size to this range (avoids FFT artifacts)
                               # set to grid_size if undesired
        self.focused_sys = False  # use this to turn scaling of beam ratio by wavelength on/off
                        # turn on/off as necessary to ensure optics in focal plane have same sampling at each
                        # wavelength. Can check focal plane sampling in the proper perscription with opx.check_sampling
                        # see Proper manual pg 36 for more info

        # Timing Params
        self.closed_loop = False  # if false (open loop), then initiate multiprocessing for individual timesteps
        self.sample_time = 0.01  # [s] seconds per timestep/frame. used in atmosphere evolution, etc
        self.ao_delay = 0  # [tstep] number of timesteps of delay
        self.startframe = 0  # useful for things like RDI
        self.numframes = 1  # number of timesteps in the simulation
        self.quick_companions = False  # this bool determines if companions are generated by simple shift and scaling
        self.quick_detect = False  # generate mkid spectral cube sequence by spatial scaling and intensity scaling datacube (no photon quantization or arteacts)

        # Plotting Params
        self.show_spectra = False  # Plot spectral cube at single timestep
        self.show_wframe = True  # Plot white light image frame
        self.show_tseries = False  # Plot full timeseries of white light frames
        self.spectra_cols = 2  # number of subplots per row in view_spectra
        self.tseries_cols = 2  # number of subplots per row in view_timeseries

        # Reading/Saving Params
        self.save_to_disk = False  # Saves observation sequence (timestep, wavelength, x, y)
        self.save_list = ['detector']  # list of locations in optics train to save
        self.skip_functions = []  # list of functions not to be applied universally by optics.Wavefronts.loop_over_function
        self.memory_limit = 10  # number of giga-bytes for sixcube of complex fields before chunking happens
        self.checkpointing = 500  # int or None number of timesteps before complex fields sixcube is saved
                                 # minimum of this and max allowed steps for memory reasons takes priority
        self.chunking = False  # chunking is neccessary if the full fields tensor does not fit in memory. RunMedis should set this automatically
        self.auto_load = False  # controls whether RunMedis checks with the user before it loads an old simulaiton it knows to have mismatching parameters
        self.degrade_photons = True

        self.verbose = True
        self.debug = False

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Astro_params:
    """
    Default parameters for the astronomical system under investigation
    exposure_time, startframe and numframes may seem a bit out of place here. Perhaps this class could be renamed
    """
    def __init__(self):
        # Companion Object Params
        self.star_flux = int(1e5)  # A 5 apparent mag star 1e6 cts/cm^2/s
        self.companion = False
        self.contrast = []
        self.C_spec = 1.5  # the gradient of the increase in contrast towards shorter wavelengths
        self.companion_xy = [[-1.0e-6, 1.0e-6]]  # [m] initial location (no rotation)
        self.spectra = [None, None]  # list of floats for Teff of Planck spectrum that scales the wavefronts

        # Wavelength and Wavefront Array Settings
        # In optics_propagate(), proper initially takes N  discreet wavelengths evenly spaced in wvl_range, where N is
        # given by ap.n_wvl_init. Later, in gen_timeseries(), the 3rd axis of the spectral cube is interpolated so that
        # there are ap.n_wvl_final over the range in ap.wvl_range.
        self.n_wvl_init = 3  # initial number of wavelength bins in spectral cube (later sampled by MKID detector)
        self.n_wvl_final = None  # final number of wavelength bins in spectral cube after interpolation (None sets equal to n_wvl_init)
        self.interp_wvl = True  # Set to interpolate wavelengths from ap.n_wvl_init to ap.n_wvl_final
        self.wvl_range = np.array([800, 1500]) / 1e9  # wavelength range in [m]
            # eg. DARKNESS band is [800, 1500], J band =  [1100,1400])

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Telescope_params:
    """
    This contains most of the parameters you will probably modify when running tests
    """
    def __init__(self):
        # Optics + Detector
        self.prescription = 'Subaru_SCExAO'#'general_telescope'#
        self.entrance_d = 5  # 7.971  # [m] telescope enterence pupil diameter in meters
        self.fnum_primary = 12  # f-number of primary
        self.flen_primary = 25  # [m] focal length of primary

        self.use_atmos = False  # Create and apply atmospheric maps (units in phase delay)
        self.obscure = False  # Obscure the primary by the secondary mirror
        self.legs_frac = 0.03  # fractional width of the legs relative to the secondary mirror diameter

        # AO System Settings
        self.use_ao = True  # if False, and DM returns an idealized 'flat'
        self.ao_act = 60  # number of actuators on the DM on one axis (proper only models nxn square DMs)
        self.piston_error = False  # flag for allowing error on DM surface shape
        self.fit_dm = True  # flag to use DM surface fitting (see proper manual pg 52, the FIT switch)
        self.satelite_speck = {'apply': False, 'phase': np.pi / 5., 'amp': 12e-9, 'xloc': 12, 'yloc': 12}

        # Ideal Detector Params (not bothering with MKIDs yet)
        #TODO Rupert should these be moved into mkid params? Are they still used?
        # self.detector = 'ideal'  # 'MKIDs'
        # self.array_size = np.array([129, 129])  # np.array([125,80])
        # self.wavecal_coeffs = [1.e9 / 12, -157]  # assume linear for now 800nm = -90deg, 1500nm = -30deg
                                                # used to make phase cubes. I assume this has something to do with the
                                                # QE of MKIDs at different wavelengths?
        self.pix_shift = [0, 0]  # False?  Shifts the central star to off-axis (mimics conex mirror, tip/tilt error)
        # self.platescale = 13.61  # mas # have to run get_sampling at the focus to find this

        # Aberrations
        self.servo_error = [0, 1]  # [0,1] # False # No delay and rate of 1/frame_time
        self.abertime = 0.5  # time scale of optic aberrations in seconds

        self.lens_params = None  # None at first then gets updated by prescription module

        # Zernike Settings- see pg 192 for details
        self.zernike_orders = [2, 3, 4]  # Order of Zernike Polynomials to include
        self.zernike_vals = np.array([175, -150, 200])*1.0e-9  # value of Zernike order in nm,
                                                               # must be same length as zernike_orders

        # Coronagraph Settings
        self.cg_type = 'Gaussian'
        self.cg_size = 3  # physical size or lambda/D size
        self.cg_size_units = "l/D"  # "m" or "l/D"
        self.lyot_size = 0.75  # units are in fraction of surface blocked
        self.fl_cg_lens = 200 * self.entrance_d  # m

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class MKID_params:
    def __init__(self):
        self.bad_pix = False
        # self.interp_sample=True # avoids the quantization error in creating the datacube
        self.QE_map = None
        self.wavecal_coeffs = [1.e9/12, -157]  # assume linear for now 800nm = -90deg, 1500nm = -30deg
        self.phase_uncertainty = False  # True
        self.phase_background = False
        self.QE_var = False
        self.remove_close = False
        self.dark_counts = False
        self.array_size = np.array([129,129])#np.array([125,80])#np.array([125,125])#
        self.resamp = True
        self.quantize_FCs = False
        # self.total_pix = self.array_size[0] * self.array_size[1]
        self.pix_yield = 0.9
        self.hot_pix = 0  # Number of hot pixels
        self.hot_bright = 1e3  # Number of counts/s a hot pixel registers
        self.dark_pix_frac = 0.1  # Number of dark count pixels
        self.dark_bright = 10  # Number of counts/s on average for dark count pixels
        self.threshold_phase = 0#-30 # quite close to 0, basically all photons will be detected.

        self.max_count = 2500.  # cts/s
        self.dead_time = 0.02#10e-6  # s
        self.bin_time = 2e-3 # minimum time to bin counts for stat-based analysis
        # self.frame_time = 0.001#atm_size*atm_spat_rate/(wind_speed*atm_scale) # 0.0004
        self.total_int = 1 #second
        self.frame_int = 1./20
        self.t_frames = int(self.total_int/self.frame_int)
        self.platescale = 10 *1e-3  # [arcsec/pix]

        # for distributions
        self.res_elements = self.array_size[0]
        self.g_mean = 0.95
        self.g_sig = 0.025
        self.g_spec = -1/10.
        self.bg_mean = 0
        self.bg_sig = 30
        self.r_mean = 1
        self.r_sig = 0.05
        self.R_mean = 50
        self.R_sig = 2
        self.R_spec = -1./10

        self.lod = 8  # 8 pixels in these upsampled images = one lambda/d
        self.nlod = 10  # 3 #how many lambda/D do we want to calculate out to

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Device_params:
    """
    This is different from MKID_params in that it contains an instance of these random multidimensional parameters
    Perhaps it could be part of MKID_params
    """
    def __init__(self):
        self.QE_map = None
        self.Rs = None
        self.sigs = None
        self.basesDeg = None
        self.hot_pix = None
        self.dark_pix_frac = None

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


class Atmos_params():
    """
    default params for making atmospheric models

    original use was to use CAOS, but has now been changed to use hcipy (2019)
    :model: 'single', 'hcipy_standard', 'evolving'
             evolving->apply variation to some parameter

    hcipy still assumes frozen flow as turbulent layers. more here: https://hcipy.readthedocs.io/en/latest/index.html
    """
    def __init__(self):
        self.model = 'single'  # single|hcipy_standard|evolving
        self.cn_sq = 0.22 * 1e-12  # magnitude of perturbations within single atmos layer, at single wavelength
        self.L0 = 10  # outer scale of the model that sets distance of layers (not boundary). used in Kalmogorov model
        self.vel = 5  # velocity of the layer in m/s
        self.h = 100  # scale height in m
        self.fried = 0.2  # m
        self.tau = 0.01 #0.1  # correlation time in seconds of atmopshere
        self.std = 2
        self.correlated_sampling = False

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]


# Creating class structures
ap = Astro_params()
tp = Telescope_params()
sp = Simulation_params()
atmp = Atmos_params()
mp = MKID_params()
dp = Device_params()
iop = IO_params()  # Must call this last since IO_params uses ap and tp

# Turning off messages from Proper
proper.print_it = False
proper.use_cubic_conv = True
# print(proper.__version__)
# proper.prop_init_savestate()

