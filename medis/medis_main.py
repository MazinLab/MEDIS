"""
run_medis_main
Kristina Davis, Rupert Dodkins

This is the version of code compiled by K. Davis and R. Dodkins updated and streamlined from the MEDISv0.0 version of
the code written by R. Dodkins during his graduate career. This code contains the same functionality as MEDISv0.0 and is
modular to continue adding features and functionality.

MEDIS is split into two main functionalities; the first is to do an end-to-end simulation of a telescope. This is more
or less a large wrapper for proper with some new capabilities built into it to enable faster system design.The second
main functionality is to convert the complex field data generated by the telescope simulator to MKID-type photon lists.

For the telescope simulator, the user builds a prescription "name.py" of the optical system of the telescope system as
a separate module in the Simulations subdirectory of MEDIS. They can then update the default params file in a "
run_name.py" type script, which also makes the call to run_MEDIS to start the simulation. The output is a 6D complex
field observation sequence.

The MKIDs camera simulator creates an instance of a device with distributions in R, dead pixel locations etc and then
combines the fields sequence with this to generate realistic detected photons

"""
import os
import numpy as np
import pickle
from datetime import datetime
from pprint import pprint

from medis.telescope import Telescope
from medis.MKIDS import Camera
from medis.params import sp, ap, tp, iop, atmp, cdip, mp

################################################################################################################
################################################################################################################
################################################################################################################
sentinel = None


class RunMedis():
    """
    Creates a simulation for calling Telescope or MKIDs to return a series of complex electric fields or photons,
    respectively.

    This class is a wrapper for Telescope and Camera that handles the checking of testdir and params existence

    Upon creation the code checks if a testdir of this name already exists, if it does it then checks if the params
    match. If the params are identical and the desired products are not already created, if it will create them.
    If the params are different or the testdir does not already exist a new testdir and simulation is created.

    """
    def __init__(self, name='test', product='fields'):
        """
        File structure:
        datadir
            testdir          <--- output
                params.pkl   <--- output

        :param name:  used at the folder name where all the data is stored
        :param product:
            fields - 6D complex tensor (t,plane,wavelength,object,x,y) product of Telescope()
            photons - photon list with non-ideal MKID affects applied
            rebinned_cube - 4D tensor (t,wavelength,x,y) like scaled_fields but with non-ideal MKID affects aplied

        """

        self.name = name
        self.product = product
        assert product in ['fields', 'photons', 'rebinned_cube'], f"Requested data product {self.product} not supported"

        iop.update_testname(self.name)
        # for storing a checking between tests
        self.params = {'ap': ap, 'tp': tp, 'atmp': atmp, 'cdip': cdip, 'iop': iop, 'sp': sp, 'mp': mp}

        # show all the parameters input into the simulation (before some are updated by Telescope and Camera)
        if sp.verbose:
            for param in self.params.values():
                print(f'\n\t {param.__name__()}')
                pprint(param.__dict__)

        # always make the top level directory if it doesn't exist yet
        if not os.path.isdir(iop.datadir):
            print(f"Top level directory... \n\n\t{iop.datadir} \n\ndoes not exist yet. Creating")
            os.makedirs(iop.datadir, exist_ok=True)

        if not os.path.exists(iop.testdir) or not os.path.exists(iop.params_logs):
            print(f"No simulation data found at... \n\n\t{iop.testdir} \n\n A new test simulation"
                  f" will be started")
            self.make_testdir()
        else:
            params_match = self.check_params()
            exact_match = all(params_match.values())
            if exact_match:
                print(f"Configuration files match. Initialization over")
            else:
                print(f"Configuration files differ")
                now = datetime.now().strftime("%m:%d:%Y_%H-%M-%S")
                backup_testr = os.path.join(iop.datadir, self.name+'_backup_'+now)
                if not sp.auto_load:
                    choice = input(f'\n\n\tINPUT REQUIRED...\n\n'
                                    f'Rename old testdir to {backup_testr} and start new simulation as {iop.testdir} [R],'
                                    f'\nor Quit [Q],'
                                    f'\nor overwrite the pickle file .pkl (keeps atmosphere and aberration maps) [P]?'
                                    f'\nor ignore parameter difference and proceed with Loading original [L]?\n')

                    if choice.lower() == 'q':
                        exit()
                    elif choice.lower() == 'r':
                        os.rename(iop.testdir, backup_testr)
                        self.make_testdir()
                    elif choice.lower() == 'p':
                        if os.path.exists(f"{iop.testdir}/fields.h5"):
                            # Don't overwrite the .pkl file if there is an existing h5 file.
                            print(f"Existing h5 file found. \n"
                                  f"Recommend saving as backup [R]\n"
                                  f"Exiting\n")
                            exit()
                        # Overwrite the .pkl files but keep aberration and atmosphere directories
                        os.remove(f"{iop.testdir}/params.pkl")

    def make_testdir(self):
        if not os.path.isdir(iop.testdir):
            os.makedirs(iop.testdir, exist_ok=True)

        with open(iop.params_logs, 'wb') as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def check_params(self):
        """ Check all param classes at this stage. Some params are updated in Telescope and Camera and those params
         should be checked against each other at that stage of the pipeline"""

        with open(iop.params_logs, 'rb') as handle:
            loaded_params = pickle.load(handle)

        match_params = {}
        print(f"\nChecking Matching Params Classes:")
        for p in ['ap','tp','atmp','cdip','iop','sp','mp']:
            matches = []
            for (this_attr, this_val), (load_attr, load_val) in zip(self.params[p].__dict__.items(),
                                                                    loaded_params[p].__dict__.items()):
                try:
                    match = this_attr == load_attr and np.all(load_val == this_val)
                except ValueError:
                    match = False

                if match == False:
                    print(f'\n\tmismatch found: this_attr= {this_attr}, load_attr= {load_attr}, this_val= {this_val}, '
                          f'load_val= {load_val}')
                matches.append(match)

            match = np.all(matches)
            print(f"param: {p}, match: {match}")
            match_params[p] = match

        return match_params

    def __call__(self, *args, **kwargs):
        """ Get fields from Telescope and optionally then get photons from Camera. This looks complicated because of
         the possibility to chunk in both Telescope and Camera but simplifies a lot if both have num_chunk = 1"""

        if self.product == 'fields':
            # get the telescope configuration
            self.tel = Telescope(usesave=sp.save_to_disk)  # checking of class's cache etc is left to the class

            #load or generate the fields
            observation = self.tel()

        else:
            self.cam = Camera(usesave=sp.save_to_disk, product=self.product)  # get the camera configuratoin
            if self.cam.photontable_exists:
                observation = self.cam()
            else:
                self.tel = Telescope(usesave=sp.save_to_disk)
                self.tel()

                if self.tel.num_chunks == 1:
                    observation = self.cam(fields=self.tel.cpx_sequence)
                else:
                    for ichunk in range(int(self.tel.num_chunks)):
                        fields = self.tel.load_fields(span=(ichunk*self.tel.chunk_steps, (ichunk+1)*self.tel.chunk_steps))['fields']

                        observation = self.cam(fields=fields, abs_step=ichunk*self.tel.chunk_steps,
                                               finalise_photontable=False)

                    if self.product == 'photons' and not self.cam.photontable_exists:
                        self.cam.save_photontable(photonlist=[], index=('ultralight', 6), populate_subsidiaries=True)
                        self.cam.photontable_exists = True
                        self.cam.save_instance()

                    print('Returning the observation data for the final chunk only')

        return observation


if __name__ == '__main__':
    sim = RunMedis(name='example1', product='photons')
    observation = sim()
    print(observation.keys())


