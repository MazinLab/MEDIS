"""
todo h5 won't let you save chunks>4GB. Add that check to max_chunk function (in addition to one users specific memory check)

"""

import os
import sys
import numpy as np
import importlib
import multiprocessing
import time
import glob
import pickle
import shutil
import tables

import proper
import medis.atmosphere as atmos
from medis.plot_tools import grid
import medis.CDI as cdi
import medis.utils as mu
import medis.optics as opx
import medis.aberrations as aber
from medis.params import sp, ap, tp, iop
from medis.CDI import cdi


class Telescope:
    """
    Creates a simulation for the telescope to create a series of complex electric fields

    During initialisation a backup of the pthon PROPER prescription is copied to the testdir, atmisphere maps and
    aberration maps are created, serialisation and memory requirements are tested, and the cdi vairable initialised

    Resulting file structure:
    datadir
        testdir
            params.pkl                     <--- input
            prescription                   <--- new
                {prescriptionname}         <--- new
                    {prescriptionname}.py  <--- new
            aberrations                    <--- new
                {aberationparams}          <--- new
                    {lensname}0.fits       <--- new
                    ...                    <--- new
            atmosphere                     <--- new
                {atmosphereparams}         <--- new
                    {atmos}0.fits          <--- new
                    ...                    <--- new
            fields.h5                      <--- output


    input
    params dict
        collection of the objects in params.py

    :return:
    self.cpx_sequence ndarray
        complex tensor of dimensions (n_timesteps, n_saved_planes, n_wavelengths, n_stars/planets, grid_size, grid_size)
    """

    def __init__(self, usesave=True):
        self.usesave=usesave

        self.fields_exists = True if os.path.exists(iop.fields) else False  # just the fields ndarray
        self.save_exists = True if os.path.exists(iop.telescope) else False  # the whole telescope object

        # if self.fields_exists:
        #     self.load_fields()
        if self.save_exists:
            print(f"\nLoading telescope instance from\n\t{iop.telescope}\n")
            with open(iop.telescope, 'rb') as handle:
                load = pickle.load(handle)
                self.__dict__ = load.__dict__  # replace the contents
                self.save_exists = True
        else:
            # copy over the prescription
            iop.prescopydir = iop.prescopydir.format(tp.prescription)
            self.target = os.path.join(iop.prescopydir, tp.prescription+'.py')

            prescriptions = iop.prescriptions_root
            fullprescription = glob.glob(os.path.join(prescriptions, '**', tp.prescription+'.py'),
                                         recursive=True)
            if len(fullprescription) == 0:
                raise FileNotFoundError
            elif len(fullprescription) > 1:
                print(f'Multiple precriptions at {fullprescription}')
                raise FileExistsError

            fullprescription = fullprescription[0]
            if os.path.exists(self.target):
                print(f'Using prescription {fullprescription}\n')
            else:
                print(f"Copying over prescription {fullprescription}\n")

                if not os.path.isdir(iop.prescopydir):
                    os.makedirs(iop.prescopydir, exist_ok=True)
                shutil.copyfile(fullprescription, self.target)

            sys.path.insert(0, os.path.dirname(fullprescription))  # load from the original prescription incase user is editting
            pres_module = importlib.import_module(tp.prescription)
            tp.__dict__.update(pres_module.tp.__dict__)  #  update tp with the contents of the prescription

            # initialize atmosphere
            iop.atmosdir = iop.atmosdir.format(sp.grid_size, sp.beam_ratio, sp.numframes)
            if glob.glob(iop.atmosdir+'/*.fits') and sp.verbose:
                print(f"Atmosphere maps already exist at \n\t{iop.atmosdir}"
                      f" \n... skipping generation\n\n")
            else:
                if not os.path.isdir(iop.atmosdir):
                    os.makedirs(iop.atmosdir, exist_ok=True)
                atmos.gen_atmos()

            # initialize aberrations
            iop.aberdir = iop.aberdir.format(sp.grid_size, sp.beam_ratio, sp.numframes)
            if glob.glob(iop.aberdir + '/*.fits') and sp.verbose:
                print(f"Aberration maps already exist at \n\t{iop.aberdir} "
                      f"\n... skipping generation\n\n")
            else:
                if not os.path.isdir(iop.aberdir):
                    os.makedirs(iop.aberdir, exist_ok=True)
                for lens in tp.lens_params:
                    aber.generate_maps(lens['aber_vals'], lens['diam'], lens['name'])

            # check if can do parrallel
            if sp.closed_loop or sp.ao_delay:
                print(f"closed loop or ao delay means sim can't be parrallelized in time domain. Forcing serial mode")
                self.parrallel = False
            else:
                self.parrallel = sp.num_processes > 1

            # ensure contrast is set properly
            if sp.quick_companions:
                # purpose of this code to add option to shift and scale an unocculed star and use that for several sources vastly decreasing compute time
                raise NotImplementedError
                ap.contrast = range(2)  # give it length two since all the planets will be collapsed into one slice

            if ap.companion is False:
                ap.contrast = []

            if ap.companion and len(ap.spectra) != len(ap.contrast)+1 != len(ap.companion_xy)+1:
                print(f'Please ensure number of sources is consistent: {len(ap.spectra)}, {len(ap.contrast)+1}, {len(ap.companion_xy)+1}')
                raise IndexError

            if not isinstance(ap.n_wvl_final, int):
                ap.n_wvl_final = ap.n_wvl_init

            # determine if can/should do all in memory
            max_steps = self.max_chunk()
            self.fieldssize = self.timestep_size * sp.numframes
            if sp.verbose: print(f"File total size should be {self.fieldssize * 1e-6} MB")
            checkpoint_steps = max_steps if sp.checkpointing is None else sp.checkpointing
            self.chunk_steps = int(min([max_steps, sp.numframes, checkpoint_steps]))
            if sp.verbose: print(f'Using time chunks of size {self.chunk_steps}')
            self.num_chunks = sp.numframes / self.chunk_steps

            if self.num_chunks > 1:
                print('Simulated data too large for dynamic memory. Storing to disk as the sim runs')
                sp.chunking = True

            self.markov = sp.chunking or self.parrallel  # independent timesteps
            nonmarkov = sp.ao_delay or sp.closed_loop  # dependent timesteps
            # if both true
            assert self.markov + nonmarkov != 2, "Confliciting modes. Request requires the timesteps be both dependent and independent"

            # neither true
            if self.markov + nonmarkov == 0:
                print("No mode specfified defaulting to markov (time independent)")
                self.markov = True

            modes = [sp.chunking, sp.ao_delay, self.parrallel,
                     sp.closed_loop]

            # Initialize List of phases of CDI probes to apply
            cdi.gen_phaseseries()
            if cdi.use_cdi:
                if hasattr(tp, 'act_tweeter'):
                    cdi.init_cout(tp.act_tweeter)
                else:
                    cdi.init_cout(tp.ao_act)

            # Remove AO planes from save_list if use_ao is False
            if not tp.use_ao and 'woofer' in sp.save_list:
                sp.save_list.remove('woofer')
            if not tp.use_ao and 'tweeter' in sp.save_list:
                sp.save_list.remove('tweeter')

    def __call__(self, *args, **kwargs):
        """ Take the observation (generate the fields sequence) """
        if self.fields_exists:
            self.load_fields()
        else:
            print('\n\n\tBeginning Telescope Simulation with MEDIS\n\n')
            start = time.time()

            self.create_fields()

            print('\n\n\tMEDIS Telescope Run Completed\n')
            finish = time.time()
            print(f'Time elapsed: {(finish - start) / 60:.2f} minutes')

            self.pretty_sequence_shape()
            self.save_exists = True

            if self.usesave and self.fieldssize > 4e9:  # bug with pickling on Mac preventing dump beyond 4GB
                print(f"\nSaving telescope instance at\n\n\t{iop.telescope}\n")
                with open(iop.telescope, 'wb') as handle:
                    pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if len(self.cpx_sequence) < sp.numframes:
            print('Returning the final time chunk instead of the full duration. Increase chunk size if you want full '
                  'duration')
        dataproduct = {'fields': np.array(self.cpx_sequence), 'sampling': self.sampling}
        return dataproduct

    def max_chunk(self):
        """
        Determines the maximum duration each chunk can be to fit within the memory limit

        :return: integer
        """
        self.timestep_size = len(sp.save_list) * ap.n_wvl_final * \
                        (1 + len(ap.contrast)) * sp.grid_size**2 * 32 / 8  # in Bytes

        max_chunk = sp.memory_limit*1e9 // self.timestep_size
        print(f'Each timestep is predicted to be {self.timestep_size/1.e6} MB, meaning no more than {max_chunk} time '
              f'steps can fit in the memory at one time')

        return max_chunk

    def create_fields(self):
        """ Create fields tensor for an initialised Telecope instance """

        t0 = sp.startframe
        self.kwargs = {}
        self.cpx_sequence = None

        if self.markov:  # time steps are independent
            ceil_num_chunks = int(np.ceil(self.num_chunks))
            if ceil_num_chunks > 1:
                print('Only partial observation will be in memory at one time')
            final_chunk_size = sp.numframes-int(np.floor(self.num_chunks))*self.chunk_steps
            for ichunk in range(ceil_num_chunks):
                fractional_step = final_chunk_size != 0 and ichunk == ceil_num_chunks-1
                chunk_steps = final_chunk_size if fractional_step else self.chunk_steps

                cpx_sequence = np.empty((chunk_steps, len(sp.save_list),
                                        ap.n_wvl_init, 1 + len(ap.contrast),
                                        sp.grid_size, sp.grid_size),
                                        dtype=np.complex64)
                chunk_range = ichunk * self.chunk_steps + t0 + np.arange(chunk_steps)
                if sp.num_processes == 1:
                    seq_samp_list = [self.run_timestep(t) for t in chunk_range]
                else:
                    print(f'Using multiprocessing of timesteps {chunk_range}')
                    # it appears as though the with statement is neccesssary when recreating Pools like this
                    with multiprocessing.Pool(processes=sp.num_processes) as p:
                        seq_samp_list = p.map(self.run_timestep, chunk_range)
                self.cpx_sequence = np.array([tup[0] for tup in seq_samp_list])
                self.sampling = seq_samp_list[0][1]

                if ap.n_wvl_init < ap.n_wvl_final:
                    self.cpx_sequence = opx.interp_wavelength(self.cpx_sequence, ax=2)
                    self.sampling = opx.interp_sampling(self.sampling)

                if sp.save_to_disk: self.save_fields(self.cpx_sequence)

        else:
            self.cpx_sequence = np.zeros((sp.numframes, len(sp.save_list),
                                          ap.n_wvl_init, 1 + len(ap.contrast),
                                          sp.grid_size, sp.grid_size), dtype=np.complex)
            self.sampling = np.zeros((len(sp.save_list), ap.n_wvl_init))

            for it, t in enumerate(range(t0, sp.numframes + t0)):
                WFS_ind = ['wfs' in plane for plane in sp.save_list]
                if t > sp.ao_delay:
                    self.kwargs['WFS_field'] = self.cpx_sequence[it - sp.ao_delay, WFS_ind, :, 0]
                    self.kwargs['AO_field'] =  self.cpx_sequence[it - sp.ao_delay, AO_ind, :, 0]
                else:
                    self.kwargs['WFS_field'] = np.zeros((ap.n_wvl_init, sp.grid_size,
                                                    sp.grid_size), dtype=np.complex)
                    self.kwargs['AO_field'] = np.zeros((ap.n_wvl_init, sp.grid_size,
                                                        sp.grid_size), dtype=np.complex)
                self.cpx_sequence[it], self.sampling = self.run_timestep(t)

            print('************************')
            if sp.save_to_disk: self.save_fields(self.cpx_sequence)

        # return {'fields': np.array(self.cpx_sequence), 'sampling': self.sampling}

    def run_timestep(self, t):
        self.kwargs['iter'] = t
        return proper.prop_run(tp.prescription, 1, sp.grid_size, PASSVALUE=self.kwargs, QUIET=True)
        # return np.zeros((1, len(sp.save_list), ap.n_wvl_init, 1 + len(ap.contrast),
        #                  sp.grid_size, sp.grid_size), dtype=np.complex64), 0

    def pretty_sequence_shape(self):

        """
        displays data format easier

        :param cpx_sequence: the 6D complex sequence generated by run_medis.telescope
        :return: nicely parsed string of 6D shape--human readable output
        """

        if len(np.shape(self.cpx_sequence)) == 6:
            samps = ['timesteps', 'save planes', 'wavelengths', 'astronomical bodies', 'x', 'y']
            delim = ', '
            print(f"Shape of cpx_sequence = " \
                f"{delim.join([samp + ':' + str(length) for samp, length in zip(samps, np.shape(self.cpx_sequence))])}")
        else:
            print(f'Warning cpx_sequence is not 6D as intented by this function. Shape of tensor ='
                  f' {self.cpx_sequence.shape}')

    def save_fields(self, fields):
        """
        Option to save fields separately from the class pickle save since fields can be huge if the user requests

        :param fields:
            dtype ndarray of complex or float
            fields can be any shape but the h5 dataset can only extended along axis 0
        :return:

        todo convert to pytables for pipeline conda integration
        """

        fields = np.array(fields)
        print(f'Saving fields dims with dimensions {fields.shape} and type {fields.dtype}')
        h5file = tables.open_file(iop.fields, mode="a", title="MEDIS Electric Fields File")
        if "/data" not in h5file:
            ds = h5file.create_earray(h5file.root, 'data', obj=fields)
        else:
            ds = h5file.root.data
            ds.append(fields)

        h5file.close()

    def load_fields(self, span=(0,-1)):
        """ load fields h5

         warning sampling is not currently stored in h5. It is stored in telescope.pkl however
         """
        print(f"Loading fields from {iop.fields}")
        h5file = tables.open_file(iop.fields, mode="r", title="MEDIS Electric Fields File")
        if span[1] == -1:
            self.cpx_sequence = h5file.root.data[span[0]:]
        else:
            self.cpx_sequence = h5file.root.data[span[0]:span[1]]
        h5file.close()
        self.pretty_sequence_shape()
        self.sampling = None

        return {'fields': self.cpx_sequence, 'sampling': self.sampling}


if __name__ == '__main__':
    iop.update_testname('telescope_module_test')

    telescope_sim = Telescope()
    observation = telescope_sim()
    print(observation.keys())
    grid(observation['fields'], logZ=True, nstd=5)
