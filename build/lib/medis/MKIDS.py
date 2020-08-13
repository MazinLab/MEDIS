"""
The Camera class simulates an obsservation with an MKID camera given a "fields" tensor (created with Telescope) and
the MKID_params that define the device
"""

import os, sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import pickle
import random
import time

from medis.distribution import *
from medis.params import mp, ap, iop, sp, tp
from medis.utils import dprint
from medis.plot_tools import view_spectra
from medis.telescope import Telescope
from medis.plot_tools import grid, quick2D


class Camera():
    def __init__(self, usesave=False, product='photons'):
        """
        Creates a simulation for the MKID Camera to create a series of photons from the fields sequence

        Resulting file structure:
        datadir
            testdir
                params.pkl         <--- input
                fields.h5          <--- input
                camera             <--- new
                    devicesave.pkl <--- new
                photonlist.h5      <--- output


        input
        fields ndarray complex
            complex tensor of dimensions (n_timesteps, n_saved_planes, n_wavelengths, n_stars/planets, grid_size, grid_size)
        usesave: bool
        product: 'photons' | 'rebinned_cube'

        :return:
        either photon list or rebinned cube

        Functions
        __init__():        loads instance of class if it exists, otherwise loads photonlist if it exist, otherwise gets
                           fields and creates device
        create_device():   create the camera's random device parameters eg locations of dead pix, assigning R to pixels etc
        __call__():        if photons can't be loaded detect photons (combine fields and device params to produce a photon
                           list) and save
        save_photonlist(): save the photonlist in the MKIDPipeline format
        save_instance():   save the whole instance of this class including photons and self.name etc

        """
        assert product in ['photons', 'rebinned_cube'], f"Requested data product {self.product} not supported"

        self.name = iop.camera  # used for saving an loading the instance
        self.usesave = usesave
        self.product = product
        self.rebinned_cube = None
        self.is_wave_cal = False

        self.save_exists = True if os.path.exists(self.name) else False  # whole Camera instance
        self.photontable_exists = True if os.path.exists(iop.photonlist) else False  # just the photonlist

        if self.save_exists and self.usesave:
            load = self.load_instance()
            self.__dict__ = load.__dict__
            self.save_exists = True  # just in case the saved obj didn't have this set to True

        else:
            # create device
            self.create_device()

    def create_device(self):
        """
        Create the camera's random device parameters eg locations of dead pix, assigning R to pixels etc

        :return:
        """
        self.platescale = mp.platescale
        self.array_size = mp.array_size
        self.dark_pix_frac = mp.dark_pix_frac
        self.hot_pix = mp.hot_pix
        self.lod = mp.lod
        self.max_count = mp.max_count
        self.numframes = sp.numframes  # sometimes numframes is different for different cams

        self.QE_map_all = self.array_QE(plot=False)
        # self.max_count = mp.max_count
        self.responsivity_error_map = self.responvisity_scaling_map(plot=False)

        if mp.bad_pix:
            self.QE_map = self.create_bad_pix(self.QE_map_all, plot=False) if mp.pix_yield != 1 else self.QE_map_all

            if mp.dark_counts:
                self.total_dark = sp.sample_time * mp.dark_bright * self.array_size[0] * self.array_size[
                    1] * self.dark_pix_frac * sp.numframes
                self.dark_locs = self.create_false_pix(amount=int(
                    mp.dark_pix_frac * self.array_size[0] * self.array_size[1]))
            if mp.hot_pix:
                self.total_hot = sp.sample_time * mp.hot_bright * self.hot_pix * sp.numframes
                self.hot_locs = self.create_false_pix(amount=mp.hot_pix)

        self.Rs = self.assign_spectral_res(plot=False)
        self.sigs = self.get_R_hyper(self.Rs, plot=False)

        # get_phase_distortions(plot=True)
        if mp.phase_background:
            self.basesDeg = self.assign_phase_background(plot=False)
        else:
            self.basesDeg = np.zeros((self.array_size))

        print('\nInitialized MKID device parameters\n')

    def __call__old(self, fields=None, abs_step=0, finalise_photontable=True, *args, **kwargs):
        if self.photontable_exists and self.usesave:
            self.load_photontable()

        else:
            assert fields is not None, 'fields must be passed in order to sample photons'

            # todo implement chunking for large datasets
            self.rebinned_cube = np.abs(np.sum(fields[:, -1], axis=2)) ** 2  # select detector plane and sum over objects
            self.rebinned_cube = self.rescale_cube(self.rebinned_cube)  # interpolate onto pixel spacing

            if sp.quick_detect and self.product == 'rebinned_cube':
                num_fake_events = int(ap.star_flux * sp.sample_time * np.sum(self.rebinned_cube))

                if sp.verbose:
                    print(f"star flux: {ap.star_flux}, cube sum: {np.sum(self.rebinned_cube)}, num fake events: {num_fake_events}")

                self.rebinned_cube *= num_fake_events
                self.photons = None
            else:
                max_steps = self.max_chunk(self.rebinned_cube)
                num_chunks = int(np.ceil(len(self.rebinned_cube)/max_steps))
                dprint(len(self.rebinned_cube), max_steps, len(self.rebinned_cube)/max_steps, num_chunks)
                self.photons = np.empty((4,0))
                for ic in range(num_chunks):
                    self.photons = self.get_photons(self.rebinned_cube[ic*max_steps:(ic+1)*max_steps],
                                                    chunk_step=abs_step + ic*max_steps)
                    if sp.degrade_photons:
                        self.photons = self.degrade_photons(self.photons)

                    if self.product == 'photons':
                        self.save_photontable(photonlist=self.photons, index=None, populate_subsidiaries=False)

                    elif self.product == 'rebinned_cube':
                        self.rebinned_cube[ic*max_steps:(ic+1)*max_steps] = self.rebin_list(self.photons, time_inds=[ic*max_steps,(ic+1)*max_steps])

                # only give the option to index and add Header after all MKID chunks are completed and even then it can be overwritten
                # for the opportunity to have chunked fields
                if finalise_photontable and self.product == 'photons':
                    self.save_photontable(photonlist=[], index=('ultralight', 6), populate_subsidiaries=True)
                    self.photontable_exists = True

            self.save_instance()

        return {'photons': self.photons, 'rebinned_cube': self.rebinned_cube}

    def quantize(self, fields, abs_step=0):
        self.rebinned_cube = np.abs(np.sum(fields[:, -1], axis=2)) ** 2  # select detector plane and sum over objects
        self.rebinned_cube = self.rescale_cube(self.rebinned_cube)  # interpolate onto pixel spacing

        if sp.quick_detect and self.product == 'rebinned_cube':
            num_fake_events = int(ap.star_flux * sp.sample_time * np.sum(self.rebinned_cube))

            if sp.verbose:
                print(f"star flux: {ap.star_flux}, cube sum: {np.sum(self.rebinned_cube)}, num fake events: {num_fake_events}")

            self.rebinned_cube *= num_fake_events
            self.photons = None

        else:
            max_steps = self.max_chunk(self.rebinned_cube)
            num_chunks = int(np.ceil(len(self.rebinned_cube) / max_steps))
            # dprint(len(self.rebinned_cube), max_steps, len(self.rebinned_cube) / max_steps, num_chunks)
            dprint(f"@Rup Please specify what you are trying to print here")
            #TODO Rupert-put a helpful print statement here instead
            self.photons = np.empty((4, 0))
            if self.product == 'photons':
                if num_chunks == 1:
                    self.photons = self.get_photons(self.rebinned_cube)
                    if self.usesave: self.save_photontable(photonlist=self.photons, index=('ultralight', 6), populate_subsidiaries=True)
                else:
                    for ic in range(num_chunks):
                        cspan = (ic * max_steps, (ic + 1) * max_steps)
                        self.photons = self.get_photons(self.rebinned_cube[cspan[0]: cspan[1]], chunk_step=abs_step + cspan[0])
                        if self.usesave: self.save_photontable(photonlist=self.photons, index=None, populate_subsidiaries=False)

                    # only give the option to index and add Header after all MKID chunks are completed and even then it
                    # can be overwritten for the opportunity to have chunked fields
                    if self.usesave:
                        self.save_photontable(photonlist=[], index=('ultralight', 6), populate_subsidiaries=True)
                        self.photontable_exists = True

                self.save_instance()
                return {'photons': self.photons}

            elif self.product == 'rebinned_cube':
                if num_chunks == 1:
                    self.photons = self.get_photons(self.rebinned_cube)
                    self.rebinned_cube = self.rebin_list(self.photons)
                else:
                    for ic in range(num_chunks):
                        cspan = (ic * max_steps, (ic + 1) * max_steps)
                        self.photons = self.get_photons(self.rebinned_cube[cspan[0]: cspan[1]], chunk_step=abs_step + cspan[0])
                        self.rebinned_cube[cspan[0]: cspan[1]] = self.rebin_list(self.photons, time_inds=[cspan[0], cspan[1]])

                self.save_instance()
                return {'rebinned_cube': self.rebinned_cube}

    def __call__(self, fields=None, abs_step=0):
        """
        Generate observation data with Camera

        Parameters
        ----------
        separate_objects: bool
            Generate photons from all targets at once or generate separate lists for each object. Separate lists are
            neccesary for training PCD algorithm
        """
        if self.photontable_exists and self.usesave:
            return self.load_photontable()

        else:
            return self.quantize(fields, abs_step)

    def max_chunk(self, datacube, round_chunk=True):
        """
        Determines the maximum duration each chunk can be to fit within the memory limit

        :return: integer
        """

        # max_counts = 50e6
        num_events = self.num_from_cube(datacube)
        self.photons_size = num_events * 32  # in Bytes (verified using photons[:,0].nbytes)

        max_chunk = sp.memory_limit*1e9 / self.photons_size
        # max_chunk = 1
        print(f'This observation will produce {num_events} photons, which is {self.photons_size/1e9} GB, meaning no '
              f'more than {max_chunk} of this observation can fit in the memory at one time')
        max_chunk = min([len(datacube)*max_chunk, len(datacube)])

        if round_chunk:
            # get nice max cut
            round_nums = [1, 2, 5]
            nice_cut = 0
            i = 0
            while nice_cut <= max_chunk:
                nice_cut = round_nums[i % len(round_nums)] * 10 ** (i // len(round_nums))  # [1,2,5,10,20,50,100,200,...]
                i += 1

            max_chunk = round_nums[(i - 2) % len(round_nums)] * 10 ** ((i - 2) // len(round_nums))  # we want the nice value thats below max_chunk so go back 2

        print(f'Input cubes will be split up into time length {max_chunk}')

        return max_chunk

    def save_photontable(self, photonlist=[], index=('ultralight', 6), timesort=False, chunkshape=None, shuffle=True, bitshuffle=False,
                        ndx_shuffle=True, ndx_bitshuffle=False, populate_subsidiaries=True):
        """
        save the photonlist in the MKIDPipeline format (https://github.com/MazinLab/MKIDPipeline)

        This code is ported from build_pytables() https://github.com/MazinLab/MKIDPipeline/blob/develop/mkidpipeline/hdf/bin2hdf.py#L64
        commit 292dec0f5140f5f1f941cc482c2fdcd2dd223011

        """
        from mkidcore.config import yaml
        from mkidcore.corelog import getLogger
        from mkidcore.headers import ObsFileCols, ObsHeader
        import mkidcore
        from mkidcore import pixelflags
        from io import StringIO
        import tables

        beammap = np.arange(mp.array_size[0]*mp.array_size[1]).reshape(mp.array_size)
        flagmap = np.zeros_like(beammap)

        h5file = tables.open_file(iop.photonlist, mode="a", title="MKID Photon File")
        filter = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=shuffle, bitshuffle=bitshuffle,
                                fletcher32=False)

        if "/Photons" not in h5file:
            group = h5file.create_group("/", 'Photons', 'Photon Information')
            table = h5file.create_table(group, name='PhotonTable', description=ObsFileCols, title="Photon Datatable",
                                        expectedrows=len(photonlist), filters=filter, chunkshape=chunkshape)
        else:
            table = h5file.root.Photons.PhotonTable

        if photonlist != []:
            xs = np.int_(photonlist[2])
            ys = np.int_(photonlist[3])
    
            ResID = beammap[xs, ys]
            photons = np.zeros(len(photonlist[0]),
                               dtype=np.dtype([('ResID', np.uint32), ('Time', np.uint32), ('Wavelength', np.float32),
                                               ('SpecWeight', np.float32), ('NoiseWeight', np.float32)]))
    
            photons['ResID'] = ResID
            photons['Time'] = photonlist[0] * 1e6 # seconds -> microseconds
            photons['Wavelength'] = self.wave_cal(photonlist[1])
            if timesort:
                photons.sort(order=('Time', 'ResID'))
                getLogger(__name__).warning('Sorting photon data on time for {}'.format(iop.photonlist))
            elif not np.all(photons['ResID'][:-1] <= photons['ResID'][1:]):
                getLogger(__name__).warning('binprocessor.extract returned data that was not sorted on ResID, sorting'
                                            '({})'.format(iop.photonlist))
                photons.sort(order=('ResID', 'Time'))


            table.append(photons)
    
            getLogger(__name__).debug('Table Populated for {}'.format(iop.photonlist))
            
        if index:
            index_filter = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=ndx_shuffle,
                                          bitshuffle=ndx_bitshuffle,
                                          fletcher32=False)

            def indexer(col, index, filter=None):
                if isinstance(index, bool):
                    col.create_csindex(filters=filter)
                else:
                    col.create_index(optlevel=index[1], kind=index[0], filters=filter)

            indexer(table.cols.Time, index, filter=index_filter)
            getLogger(__name__).debug('Time Indexed for {}'.format(iop.photonlist))
            indexer(table.cols.ResID, index, filter=index_filter)
            getLogger(__name__).debug('ResID Indexed for {}'.format(iop.photonlist))
            indexer(table.cols.Wavelength, index, filter=index_filter)
            getLogger(__name__).debug('Wavelength indexed for {}'.format(iop.photonlist))
            getLogger(__name__).debug('Table indexed ({}) for {}'.format(index, iop.photonlist))
            print('Table indexed ({}) for {}'.format(index, iop.photonlist))
        else:
            getLogger(__name__).debug('Skipping Index Generation for {}'.format(iop.photonlist))

        if populate_subsidiaries:
            group = h5file.create_group("/", 'BeamMap', 'Beammap Information', filters=filter)
            h5file.create_array(group, 'Map', beammap, 'resID map')
            h5file.create_array(group, 'Flag', flagmap, 'flag map')
            getLogger(__name__).debug('Beammap Attached to {}'.format(iop.photonlist))

            h5file.create_group('/', 'header', 'Header')
            headerTable = h5file.create_table('/header', 'header', ObsHeader, 'Header')
            headerContents = headerTable.row
            headerContents['isWvlCalibrated'] = True
            headerContents['isFlatCalibrated'] = True
            headerContents['isSpecCalibrated'] = True
            headerContents['isLinearityCorrected'] = True
            headerContents['isPhaseNoiseCorrected'] = True
            headerContents['isPhotonTailCorrected'] = True
            headerContents['timeMaskExists'] = False
            headerContents['startTime'] = int(time.time())
            headerContents['expTime'] = np.ceil(sp.sample_time * sp.numframes)
            headerContents['wvlBinStart'] = ap.wvl_range[0] * 1e9
            headerContents['wvlBinEnd'] = ap.wvl_range[1] * 1e9
            headerContents['energyBinWidth'] = 0.1  #todo check this
            headerContents['target'] = ''
            headerContents['dataDir'] = iop.testdir
            headerContents['beammapFile'] = ''
            headerContents['wvlCalFile'] = ''
            headerContents['fltCalFile'] = ''
            headerContents['metadata'] = ''

            out = StringIO()
            yaml.dump({'flags': mkidcore.pixelflags.FLAG_LIST}, out)
            out = out.getvalue().encode()
            if len(out) > mkidcore.headers.METADATA_BLOCK_BYTES:  # this should match mkidcore.headers.ObsHeader.metadata
                raise ValueError("Too much metadata! {} KB needed, {} allocated".format(len(out) // 1024,
                                                                                        mkidcore.headers.METADATA_BLOCK_BYTES // 1024))
            headerContents['metadata'] = out

            headerContents.append()
            getLogger(__name__).debug('Header Attached to {}'.format(iop.photonlist))

        h5file.close()
        getLogger(__name__).debug('Done with {}'.format(iop.photonlist))

        # self.photontable_exists = True

    def load_photontable(self):
        """Load photon list from pipeline's photon table h5"""
        import tables
        h5file = tables.open_file(iop.photonlist, "r")
        table = h5file.root.Photons.PhotonTable
        self.is_wave_cal = h5file.root.header.header.description.isWvlCalibrated

        self.photons = np.zeros((4, len(table)))
        self.photons[0] = table[:]['Time'] /1e6
        self.photons[1] = table[:]['Wavelength']

        rev_beam = []
        [[rev_beam.append([c,r]) for r in range(mp.array_size[1])] for c in range(mp.array_size[0])]
        rev_beam = np.array(rev_beam)
        self.photons[2:] = (rev_beam[table[:]['ResID']].T)
        self.rebinned_cube = None
        print(f"Loaded photon list from table at{h5file}")

        return {'photons': self.photons}

    def save_instance(self):
        """
        Save the whole Camera instance

        This is a defensive way to write pickle.write, allowing for very large files on all platforms
        """
        max_bytes = 2 ** 31 - 1
        bytes_out = pickle.dumps(self)
        n_bytes = sys.getsizeof(bytes_out)
        with open(self.name, 'wb') as f_out:
            for idx in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[idx:idx + max_bytes])

    def load_instance(self):
        """
        This is a defensive way to write pickle.load, allowing for very large files on all platforms
        """
        max_bytes = 2 ** 31 - 1
        try:
            input_size = os.path.getsize(self.name)
            bytes_in = bytearray(0)
            with open(self.name, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            obj = pickle.loads(bytes_in)
        except:
            return None
        return obj

    def num_from_cube(self, datacube):
        return int(ap.star_flux * sp.sample_time * np.sum(datacube))

    def get_photons(self, datacube, chunk_step=0, plot=False):
        """
        Given an intensity spectralcube and timestep create a quantized photon list

        :param datacube:
        :param step:
        :param plot:
        :return:
        """

        if mp.QE_var:
            datacube *= self.QE_map.T

        num_events = self.num_from_cube(datacube)

        if sp.verbose: print(f"star flux: {ap.star_flux}, cube sum: {np.sum(datacube)}, numframes: {self.numframes},"
              f"num events: {num_events:e}")

        photons = self.sample_cube(datacube, num_events)
        photons[0] += chunk_step
        photons[0] = self.assign_time(photons[0])
        photons[1] = self.assign_phase(photons[1])

        if plot:
            grid(self.rebin_list(photons), title='get_photons')

        if sp.degrade_photons:
            photons = self.degrade_photons(photons)

        return photons

    def degrade_photons(self, photons, plot=False):
        if plot:
            grid(self.rebin_list(photons), title='before degrade')

        if mp.dark_counts:
            dark_photons = self.get_bad_packets(type='dark')
            dprint(photons.shape, dark_photons.shape, 'dark')
            photons = np.hstack((photons, dark_photons))

        if mp.hot_pix:
            hot_photons = self.get_bad_packets(type='hot')
            photons = np.hstack((photons, hot_photons))
            # stem = add_hot(stem)

        if plot:
            grid(self.rebin_list(photons), title='after bad')

        if mp.phase_uncertainty:
            photons[1] *= self.responsivity_error_map[np.int_(photons[2]), np.int_(photons[3])]
            photons, idx = self.apply_phase_offset_array(photons, self.sigs)

        # thresh =  photons[1] < self.basesDeg[np.int_(photons[3]),np.int_(photons[2])]
        if mp.phase_background:
            thresh =  -photons[1] > 3*self.sigs[-1,np.int_(photons[3]), np.int_(photons[2])]
            photons = photons[:, thresh]

        if mp.remove_close:
            stem = self.arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
            stem = self.remove_close(stem)
            photons = self.ungroup(stem)

        if plot:
            grid(self.rebin_list(photons), title='after remove close')

        # This step was taking a long time
        # stem = arange_into_stem(photons.T, (self.array_size[0], self.array_size[1]))
        # cube = make_datacube(stem, (self.array_size[0], self.array_size[1], ap.n_wvl_final))
        # # ax7.imshow(cube[0], origin='lower', norm=LogNorm(), cmap='inferno', vmin=1)
        # cube /= self.QE_map
        # photons = ungroup(stem)

        return photons

    def arange_into_cube(self, packets, size):
        # print 'Sorting packets into xy grid (no phase or time sorting)'
        cube = [[[] for i in range(size[0])] for j in range(size[1])]
        for ip, p in enumerate(packets):
            x = np.int_(p[2])
            y = np.int_(p[3])
            cube[x][y].append([p[0] ,p[1]])
            if len(packets) >= 1e7 and ip % 1000:
                misc.progressBar(value=ip, endvalue=len(packets))
        # print cube[x][y]
        # cube = time_sort(cube)
        return cube

    def responvisity_scaling_map(self, plot=False):
        """Assigns each pixel a phase responsivity between 0 and 1"""
        dist = Distribution(gaussian(mp.r_mean, mp.r_sig, np.linspace(0, 2, mp.res_elements)), interpolation=True)
        responsivity = dist(self.array_size[0] * self.array_size[1])[0]/float(mp.res_elements) * 2
        if plot:
            plt.xlabel('Responsivity')
            plt.ylabel('#')
            plt.hist(responsivity)
            plt.show()
        responsivity = np.reshape(responsivity, self.array_size)
        if plot:
            quick2D(responsivity)#plt.imshow(QE)
            # plt.show()

        return responsivity

    def array_QE(self, plot=False):
        """Assigns each pixel a phase responsivity between 0 and 1"""
        dist = Distribution(gaussian(mp.g_mean, mp.g_sig, np.linspace(0, 1, mp.res_elements)), interpolation=True)
        QE = dist(self.array_size[0] * self.array_size[1])[0]/float(mp.res_elements)
        if plot:
            plt.xlabel('Responsivity')
            plt.ylabel('#')
            plt.hist(QE)
            plt.show()
        QE = np.reshape(QE, self.array_size[::-1])
        if plot:
            quick2D(QE)#plt.imshow(QE)
            # plt.show()

        return QE

    def assign_spectral_res(self, plot=False):
        """Assigning each pixel a spectral resolution (at 800nm)"""
        dist = Distribution(gaussian(0.5, 0.25, np.linspace(-0.2, 1.2, mp.res_elements)), interpolation=True)
        # print(f"Mean R = {mp.R_mean}")
        Rs = (dist(self.array_size[0]*self.array_size[1])[0]/float(mp.res_elements)-0.5)*mp.R_sig + mp.R_mean#
        if plot:
            plt.xlabel('R')
            plt.ylabel('#')
            plt.hist(Rs)
            plt.show()
        Rs = np.reshape(Rs, self.array_size)

        if plot:
            plt.figure()
            plt.imshow(Rs)
            plt.show()
        return Rs

    def get_R_hyper(self, Rs, plot=False):
        """Each pixel of the array has a matrix of probabilities that depends on the input wavelength"""
        print('Creating a cube of R standard deviations')
        m = (mp.R_spec*Rs)/(ap.wvl_range[1] - ap.wvl_range[0]) # looses R of 10% over the 700 wvl_range
        c = Rs-m*ap.wvl_range[0] # each c depends on the R @ 800
        waves = np.ones((np.shape(m)[1],np.shape(m)[0],ap.n_wvl_final+5))*np.linspace(ap.wvl_range[0],ap.wvl_range[1],ap.n_wvl_final+5)
        waves = np.transpose(waves) # make a tensor of 128x128x10 where every 10 vector is 800... 1500
        R_spec = m * waves + c # 128x128x10 tensor is now lots of simple linear lines e.g. 50,49,.. 45
        # probs = np.ones((np.shape(R_spec)[0],np.shape(R_spec)[1],np.shape(R_spec)[2],
        #                 mp.res_elements))*np.linspace(0, 1, mp.res_elements)
        #                         # similar to waves but 0... 1 using 128 elements
        # R_spec = np.repeat(R_spec[:,:,:,np.newaxis], mp.res_elements, 3) # creat 128 repeats of R_spec so (10,128,128,128)
        # mp.R_probs = gaussian(0.5, R_spec, probs) #each xylocation is gaussian that gets wider for longer wavelengths
        sigs_w = (waves/R_spec)/2.35 #R = w/dw & FWHM = 2.35*sig

        # plt.plot(range(0,1500),spec.phase_cal(np.arange(0,1500)))
        # plt.show()
        sigs_p = self.phase_cal(sigs_w) - self.phase_cal(np.zeros_like((sigs_w)))

        if plot:
            plt.figure()
            plt.plot(R_spec[:, 0, 0])
            plt.plot(R_spec[:,50,15])
            # plt.figure()
            # plt.plot(sigs_w[:,0,0])
            # plt.plot(sigs_w[:,50,15])
            # plt.figure()
            # plt.plot(sigs_p[:, 0, 0])
            # plt.plot(sigs_p[:, 50, 15])
            # plt.figure()
            # plt.imshow(sigs_p[:,:,0], aspect='auto')
            view_spectra(sigs_w, show=False)
            view_spectra(sigs_p, show=False)
            plt.figure()
            plt.plot(np.mean(sigs_w, axis=(1,2)))
            plt.figure()
            plt.plot(np.mean(sigs_p, axis=(1,2)))
            # plt.imshow(mp.R_probs[:,0,0,:])
            plt.show(block=True)
        return sigs_p

    def apply_phase_scaling(photons, ):
        """
        From things like resonator Q, bias power, quasiparticle losses

        :param photons:
        :return:
        """

    def apply_phase_offset_array(self, photons, sigs):
        """
        From things like IQ phase offset noise

        :param photons:
        :param sigs:
        :return:
        """
        wavelength = self.wave_cal(photons[1])

        idx = self.wave_idx(wavelength)

        bad = np.where(np.logical_or(idx>=len(sigs), idx<0))[0]

        photons = np.delete(photons, bad, axis=1)
        idx = np.delete(idx, bad)

        distortion = np.random.normal(np.zeros((photons[1].shape[0])),
                                      sigs[idx,np.int_(photons[2]), np.int_(photons[3])])

        photons[1] += distortion

        return photons, idx

    def apply_phase_distort(self, phase, loc, sigs):
        """
        Simulates phase height of a real detector system per photon
        proper will spit out the true phase of each photon it propagates. this function will give it
        a 'measured' phase based on the resolution of the detector, and a Gaussian distribution around
        the center of the resolution bin

        :param phase: real(exact) phase information from Proper
        :param loc:
        :param sigs:
        :return: distorted phase
        """
        # phase = phase + mp.phase_distortions[ip]
        wavelength = self.wave_cal(phase)
        idx = self.wave_idx(wavelength)

        if phase != 0 and idx<len(sigs):
            phase = np.random.normal(phase,sigs[idx,loc[0],loc[1]],1)[0]
        return phase

    def wave_idx(self, wavelength):
        m = float(ap.n_wvl_final-1)/(ap.wvl_range[1] - ap.wvl_range[0])
        c = -m*ap.wvl_range[0]
        idx = wavelength*m + c
        # return np.int_(idx)
        return np.int_(np.round(idx))

    def phase_cal(self, wavelengths):
        '''Wavelength in nm'''
        phase = mp.wavecal_coeffs[0]*wavelengths + mp.wavecal_coeffs[1]
        return phase

    def wave_cal(self, phase):
        wave = (phase - mp.wavecal_coeffs[1])/(mp.wavecal_coeffs[0])
        return wave

    def assign_phase_background(self, plot=False):
        """assigns each pixel a baseline phase"""
        dist = Distribution(gaussian(0.5, 0.25, np.linspace(-0.2, 1.2, mp.res_elements)), interpolation=True)

        basesDeg = dist(self.array_size[0]*self.array_size[1])[0]/float(mp.res_elements)*mp.bg_mean/mp.g_mean
        if plot:
            plt.xlabel('basesDeg')
            plt.ylabel('#')
            plt.title('Background Phase')
            plt.hist(basesDeg)
            plt.show(block=True)
        basesDeg = np.reshape(basesDeg, self.array_size)
        if plot:
            plt.title('Background Phase--Reshaped')
            plt.imshow(basesDeg)
            plt.show(block=True)
        return basesDeg


    def create_bad_pix(self, QE_map, plot=False):
        amount = int(self.array_size[0]*self.array_size[1]*(1.-mp.pix_yield))

        bad_ind = random.sample(list(range(self.array_size[0]*self.array_size[1])), amount)

        dprint(f"Bad indices = {len(bad_ind)}, # MKID pix = { self.array_size[0]*self.array_size[1]}, "
               f"Pixel Yield = {mp.pix_yield}, amount? = {amount}")

        # bad_y = random.sample(y, amount)
        bad_y = np.int_(np.floor(bad_ind/self.array_size[1]))
        bad_x = bad_ind % self.array_size[1]

        # print(f"responsivity shape  = {responsivities.shape}")
        QE_map = np.array(QE_map)

        QE_map[bad_x, bad_y] = 0

        if plot:
            plt.xlabel('x pix')
            plt.ylabel('y pix')
            plt.title('Bad Pixel Map')
            plt.imshow(QE_map)
            cax = plt.colorbar()
            cax.set_label('Responsivity')
            plt.show()

        return QE_map


    def create_bad_pix_center(self, responsivities):
        res_elements=self.array_size[0]
        # responsivities = np.zeros()
        for x in range(self.array_size[1]):
            dist = Distribution(gaussian(0.5, 0.25, np.linspace(0, 1, mp.res_elements)), interpolation=False)
            dist = np.int_(dist(int(self.array_size[0]*mp.pix_yield))[0])#/float(mp.res_elements)*np.int_(self.array_size[0]) / mp.g_mean)
            # plt.plot(dist)
            # plt.show()
            dead_ind = []
            [dead_ind.append(el) for el in range(self.array_size[0]) if el not in dist]
            responsivities[x][dead_ind] = 0

        return responsivities

    def get_bad_packets(self, type='dark'):
        if type == 'hot':
            n_device_counts = self.total_hot
        elif type == 'dark':
            n_device_counts = self.total_dark
            dprint(self.total_dark, mp.dark_bright, sp.sample_time)
        else:
            print("type currently has to be 'hot' or 'dark'")
            raise AttributeError

        if n_device_counts % 1 > np.random.uniform(0,1,1):
            n_device_counts += 1

        n_device_counts = int(n_device_counts)
        photons = np.zeros((4, n_device_counts))
        if n_device_counts > 0:
            if type == 'hot':
                phases = np.random.uniform(-120, 0, n_device_counts)
                hot_ind = np.random.choice(range(len(self.hot_locs[0])), n_device_counts)
                bad_pix = self.hot_locs[:, hot_ind]
            elif type == 'dark':
                dist = Distribution(gaussian(0, 0.25, np.linspace(0, 1, mp.res_elements)), interpolation=False)
                phases = dist(n_device_counts)[0]
                max_phase = max(phases)
                phases = -phases*120/max_phase
                bad_pix_options = self.create_false_pix(self.dark_pix_frac * self.array_size[0] * self.array_size[1])
                bad_ind = np.random.choice(range(len(bad_pix_options[0])), n_device_counts)
                bad_pix = bad_pix_options[:, bad_ind]

            photons[0] = np.random.uniform(sp.startframe * sp.sample_time, sp.numframes * sp.sample_time, n_device_counts)
            photons[1] = phases
            photons[2:] = bad_pix

        return photons

    def create_false_pix(self, amount):
        # print(f"amount = {amount}")
        bad_ind = random.sample(list(range(self.array_size[0]*self.array_size[1])), int(amount))
        bad_y = np.int_(np.floor(bad_ind / self.array_size[1]))
        bad_x = bad_ind % self.array_size[1]

        return np.array([bad_x, bad_y])


    def remove_bad(self, frame, QE):
        bad_map = np.ones((sp.grid_size,sp.grid_size))
        bad_map[QE[:-1,:-1]==0] = 0
        # quick2D(QE, logZ =False)
        # quick2D(bad_map, logZ =False)
        frame = frame*bad_map
        return frame

    def sample_cube(self, datacube, num_events):
        # print 'creating photon data from reference cube'

        dist = Distribution(datacube, interpolation=2)

        photons = dist(num_events)
        return photons

    def assign_calibtime(self, photons, step):
        meantime = step*sp.sample_time
        # photons = photons.astype(float)#np.asarray(photons[0], dtype=np.float64)
        # photons[0] = photons[0] * ps.mp.frame_time
        timedist = np.random.uniform(meantime-sp.sample_time/2, meantime+sp.sample_time/2, len(photons[0]))
        photons = np.vstack((timedist, photons))
        return photons

    def assign_time(self, time_idx):
        return time_idx*sp.sample_time

    def assign_phase(self, phase_idx):
        """
        idx -> phase

        :parameter: idx (list)

        :return: phase (list)
        """
        phase_idx = np.array(phase_idx)
        c = ap.wvl_range[0]
        m = (ap.wvl_range[1] - ap.wvl_range[0])/ap.n_wvl_final
        wavelengths = phase_idx*m + c

        return self.phase_cal(wavelengths)

    def make_datacube_from_list(self, packets):
        phase_band = self.phase_cal(ap.wvl_range)
        bins = [np.linspace(phase_band[0], phase_band[1], ap.n_wvl_final + 1),
                range(self.array_size[0]+1),
                range(self.array_size[1]+1)]
        datacube, _ = np.histogramdd(packets[:,1:], bins)

        return datacube

    def rebin_list(self, photons, time_inds=None):
        phase_band = self.phase_cal(ap.wvl_range)
        if not time_inds:
            time_inds = [0, self.numframes]
        bins = [np.linspace(sp.sample_time * time_inds[0], sp.sample_time * time_inds[1], time_inds[1] - time_inds[0] + 1),
                np.linspace(phase_band[0], phase_band[1], ap.n_wvl_final + 1),
                range(self.array_size[0] + 1),
                range(self.array_size[1] + 1)]
        if self.is_wave_cal:
            bins[1] = self.wave_cal(np.linspace(phase_band[0], phase_band[1], ap.n_wvl_final + 1))
        rebinned_cube, _ = np.histogramdd(photons.T, bins)
        return rebinned_cube

    def cut_max_count(self, datacube):
        image = np.sum(datacube, axis=0)
        max_counts = self.max_count * sp.sample_time
        maxed = image > max_counts
        scaling_map = max_counts / image
        scaling_map[scaling_map > 1] = 1
        scaled_cube = datacube * scaling_map
        dprint(f'datacube went from {np.sum(datacube)} to {np.sum(scaled_cube)} '
               f'(factor of {np.sum(scaled_cube)/np.sum(datacube)}) during max count cut')

        return scaled_cube

    def remove_close(self, stem):
        print('removing close photons')
        for x in range(len(stem)):
            for y in range(len(stem[0])):
                # print(x,y)
                if len(stem[x][y]) > 1:
                    events = np.array(stem[x][y])
                    timesort = np.argsort(events[:, 0])
                    events = events[timesort]
                    detected = [0]
                    idx = 0
                    while idx != None:
                        these_times = events[idx:, 0] - events[idx, 0]
                        detect, _ = next(((i, v) for (i, v) in enumerate(these_times) if v > mp.dead_time),
                                         (None, None))
                        if detect != None:
                            detect += idx
                            detected.append(detect)
                        idx = detect

                    missed = [ele for ele in range(detected[-1] + 1) if ele not in detected]
                    events = np.delete(events, missed, axis=0)
                    stem[x][y] = events
                    # dprint(x, len(missed))
        return stem

    def arange_into_stem(self, packets, size):
        # print 'Sorting packets into xy grid (no phase or time sorting)'
        stem = [[[] for i in range(size[0])] for j in range(size[1])]
        # dprint(np.shape(cube))
        # plt.hist(packets[:,1], bins=100)
        # plt.show()
        for ip, p in enumerate(packets):
            x = np.int_(p[2])
            y = np.int_(p[3])
            stem[x][y].append([p[0], p[1]])
            if len(packets) >= 1e7 and ip % 10000 == 0: misc.progressBar(value=ip, endvalue=len(packets))
        # print cube[x][y]
        # cube = time_sort(cube)
        return stem

    def ungroup(self, stem):
        photons = np.empty((0, 4))
        for x in range(mp.array_size[1]):
            for y in range(mp.array_size[0]):
                # print(x, y)
                if len(stem[x][y]) > 0:
                    events = np.array(stem[x][y])
                    xy = [[x, y]] * len(events) if len(events.shape) == 2 else [x, y]
                    events = np.append(events, xy, axis=1)
                    photons = np.vstack((photons, events))
                    # print(events, np.shape(events))
                    # timesort = np.argsort(events[:, 0])
                    # events = events[timesort]
                    # sep = events[:, 0] - np.roll(events[:, 0], 1, 0)
        return photons.T

    def get_ideal_photons(self, cube, step):
        raise NotImplementedError
        ncounts = np.sum(cube)
        photons = np.zeros((ncounts, 4))
        photons[:,0] = step*sp.sample_time
        for iw, slice in enumerate(cube):
            nslicecounts = np.sum(cube)
            photons[:, iw*nslicecounts:(iw+1)*nslicecounts] = ap.wvl_samp[iw]
            for x in mp.array_size[0]:
                for y in mp.array_size[1]:
                    npixcounts = slice[x,y]
                    photons[:,:,x*npixcounts:(x+1)*npixcounts,y*npixcounts:(y+1)*npixcounts] = [x,y]

    def rescale_cube(self, rebinned_cube, conserve=True):
        if conserve:
            total = np.sum(rebinned_cube)
        nyq_sampling = ap.wvl_range[0]*360*3600/(2*np.pi*tp.entrance_d)
        self.sampling = nyq_sampling*sp.beam_ratio*2  # nyq sampling happens at sp.beam_ratio = 0.5
        x = np.arange(-sp.grid_size*self.sampling/2, sp.grid_size*self.sampling/2, self.sampling)
        xnew = np.arange(-self.array_size[0]*self.platescale/2, self.array_size[0]*self.platescale/2, self.platescale)
        ynew = np.arange(-self.array_size[1]*self.platescale/2, self.array_size[1]*self.platescale/2, self.platescale)
        mkid_cube = np.zeros((rebinned_cube.shape[0], rebinned_cube.shape[1], self.array_size[0], self.array_size[1]))
        for d, datacube in enumerate(rebinned_cube):
            for s, slice in enumerate(datacube):
                f = interpolate.interp2d(x, x, slice, kind='cubic')
                mkid_cube[d, s] = f(ynew, xnew)
        mkid_cube = mkid_cube*np.sum(datacube)/np.sum(mkid_cube)
        # grid(mkid_cube, logZ=True, show=True, extract_center=False, title='post')
        rebinned_cube = mkid_cube

        rebinned_cube[rebinned_cube < 0] *= -1

        if conserve:
            rebinned_cube *= total/np.sum(rebinned_cube)

        return rebinned_cube


if __name__ == '__main__':
    iop.update_testname('MKIDS_module_test')
    sp.quick_detect = True

    cam = Camera()
    observation = cam()
    print(observation.keys())
    grid(observation['rebinned_photons'], nstd=5)
