import os
import numpy as np
from subprocess import Popen, PIPE
import f90nml
import pickle
import datetime
import warnings


class AstraCoreError(Exception):
    """ AstraCore error class.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class AstraCoreWarning(UserWarning):
    """ AstraCore warning class.
    """
    pass


class PostProcessError(Exception):
    """ Post Process error class.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class PostProcessWarning(UserWarning):
    """ Post Process warning class.
    """
    pass


class AstraCore:
    """ A tool that eases the painful scans in ASTRA simulations.
    """

    def __init__(self, root):
        """ Initialize the Astra commander by passing a simulation directory to it.

        Keyword arguments:
        root -- the path to the root simulation folder.
        """
        self.log = []
        self.root = root
        self.dist = None
        self.main = None
        self._get_project_info()

    @staticmethod
    def modify(fname, patch=None, path='', debug=1):
        """ Modify the astra input file by a patch.
        Cannot add new namelists yet.

        Keyword arguments:
        fname -- filename with extension.
        patch -- [None] the patch applied to the file.
        path -- [''] the path to the input file.
        debug -- [1] if active the debug mode.
            Astra has a bug in reading f90 namelists. If the bug has been addressed,
            this switch should be turned off.

        Returns:
        R -- description.
        """
        fullname = os.path.join(path, fname)
        nml = f90nml.read(fullname)

        try:
            for key in patch.keys():
                _key = key.lower()
                if _key in nml.keys():
                    for subkey in patch[key].keys():
                        _subkey = subkey.lower()
                        nml[_key][_subkey] = patch[key][subkey]
                        if debug:
                            if _key == 'cavity' and _subkey == 'phi':
                                phi_list = patch[key][subkey]
                                _phi_list = []
                                try:
                                    last_phi = phi_list[-1]
                                    _phi_list.append(last_phi)
                                    for i in range(len(phi_list) - 1):
                                        _phi_list.append(phi_list[i] - last_phi)
                                    nml[_key][_subkey] = _phi_list
                                except:
                                    pass

            nml.write(fullname, True)
        except AttributeError:
            warnings.warn("nothing changed.", AstraCoreWarning)
        except:
            raise

    @staticmethod
    def get_output(out):
        """ Decode the astra output to get a unicode output.

        Keyword arguments:
        out -- astra output.

        Returns:
        output -- decoded output.
        """
        out = out[0].decode(encoding='UTF-8')

        return out

    @staticmethod
    def get_run_time(out):
        """ Extract the running time out of the astra output.

        Keyword arguments:
        out -- astra output.

        Returns:
        time -- running time in second.
        """
        out = out[0].decode(encoding='UTF-8')

        try:
            time_line = [l for l in out.split('\n') if
                         l.strip().startswith('execution time')][0]
            return float(time_line.split(':')[1].strip().split()[0])
        except:
            raise

    @staticmethod
    def is_output(fname):
        """ Test if a filename is a Astra output file.

        Keyword arguments:
        fname -- filename without path.

        Returns:
        isout -- boolean.
        """
        isout = False
        try:
            pos, rnum = fname.split('.')[1:]
            isout = pos.isdigit() and rnum.isdigit()
        except:
            pass

        return isout

    def _get_project_info(self):
        """ Get project info by scanning the simulation folder.

        Fills in self.dist and self.main. Cannot decide which one is the correct one if there
        are multiple input files.
        """
        root = self.root
        input_fnames = [f for f in os.listdir(root) if os.path.splitext(f)[1] == '.in']

        for fname in input_fnames:
            fullname = os.path.join(root, fname)
            nml = f90nml.read(fullname)

            if 'newrun' in nml.keys():
                self.main = fname
            elif 'input' in nml.keys():
                self.dist = fname

    def peek(self, fname=None):
        """ Peek into the content of the input file.

        Keyword arguments:
        fname -- [None] filename of the astra input file with extension.
            if None, peek into the self.main file.

        Returns:
        nml -- a f90 namelist contains all the infomations.
        """
        if fname is None:
            fname = self.main
        fullname = os.path.join(self.root, fname)
        nml = f90nml.read(fullname)

        return nml

    def run(self, patch=None, sim='000', show=False):
        """ Run the simulation. Support both win and mac.

        Keyword arguments:
        patch -- [None] patch that applied to the original input files.
            Could be dist patch or main patch or a mix of both.
        sim -- ['000'] the folder name in which to run the simulation.
        show -- [False] if show the output of astra.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cmd = ['mkdir', sim]

        mk = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=self.root, shell=(os.name != 'posix'))
        out = mk.communicate()

        sim_path = os.path.join(self.root, sim)
        if os.name == 'posix':
            cmd = 'rm -f *'
        else:
            cmd = 'del /s/q/f *'

        rm = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=sim_path, shell=(os.name != 'posix'))
        out = rm.communicate()

        if os.name == 'posix':
            cmd = ['find', '.', '-maxdepth', '1', '-type', 'f', '-exec', 'cp', '{}', sim, ';']
        else:
            cmd = 'xcopy . {} /y'.format(sim)

        cp = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=self.root)
        out = cp.communicate()

        if self.dist is None:
            raise AstraCoreError(
                'please indicate the beam distribution input file!')
        if patch:
            self.modify(self.dist, patch, sim_path)

        cmd = ['generator', self.dist]

        gen = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=sim_path)
        out = gen.communicate()

        if self.main is None:
            raise AstraCoreError(
                'please indicate the main input file!')
        if patch:
            self.modify(self.main, patch, sim_path)

        cmd = ['Astra', self.main]

        do = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=sim_path)
        out = do.communicate()

        # Log the simulation result
        self.log.append([timestamp, sim, patch, out])

        if show:
            print(self.get_output(out))

    @staticmethod
    def filter_data(data, kind='g'):
        """ Filter the data by the flag column in data

        Keyword arguments:
        data -- x, y, z, px, py, pz, t.
            x, y, z -- mm.
            px, py, pz -- MeV/c.
            t -- ps.
        kind -- ['g'] the filter keyword.
            'g' -- the good particles.
            'f' -- the fallback particles.
            'l' -- the particles loss along the beamline.
            'c' -- the particles loss at cathode.
            'b' -- the bad particles.
            None -- don't apply the filter.

        Returns:
        data -- the filtered data.
        """
        try:
            flag = data[7]

            if kind == 'g':  # good
                return data[:7, ((flag == 3) | (flag == 5))]
            elif kind == 'f':  # fallback
                return data[:7, ((flag == -89) | (flag == -90))]
            elif kind == 'l':  # loss
                return data[:7, ((flag == -15) | (flag == -17))]
            elif kind == 'c':  # cathode
                return data[:7, ((flag == -1) | (flag == -3))]
            elif kind == 'b':  # bad
                return data[:7, flag < 0]
            elif kind is None:  # don't apply
                return data
            else:
                msg = 'the filter keyword {} is not supported yet, ' \
                      'the original data is returned!'.format(kind)
                warnings.warn(msg, AstraCoreWarning)
                return data
        except IndexError:
            warnings.warn("the data doesn't have a flag column, "
                          "the original data is returned!", AstraCoreWarning)
            return data

    def get_data_by_name(self, sim, fname, kind=None):
        """ Read astra data file like 'linac.0011.001' and get all the infomation.

        Keyword arguments:
        sim -- simulation folder in which to find the output file.
        fname -- filename of the output file.
        kind -- [None] the filter keyword.
            None -- don't apply the filter.
            'g' -- the good particles.
            'f' -- the fallback particles.
            'l' -- the particles loss along the beamline.
            'c' -- the particles loss at cathode.
            'b' -- the bad particles.

        Returns:
        data -- the information read.
        """
        fullname = os.path.join(self.root, sim, fname)
        data = np.loadtxt(fullname)

        ref = data[0]  # modify the data based on the reference particle
        data[1:, :7] += ref[:7]

        x = data[:, 0] * 1e3  # change unit from m to mm
        y = data[:, 1] * 1e3
        z = data[:, 2] * 1e3
        px = data[:, 3] * 1e-6  # transfer the unit of p from eV/c to MeV/c
        py = data[:, 4] * 1e-6
        pz = data[:, 5] * 1e-6
        t = data[:, 6] * 1e3  # ns to ps
        flag = data[:, 9].astype(int)

        data = np.array([x, y, z, px, py, pz, t, flag])
        data = self.filter_data(data, kind)

        return data

    def get_data(self, sim, idx=0, kind=None):
        """ Read astra data file like 'linac.0011.001' and get all the infomation.

        Keyword arguments:
        sim -- simulation folder in which to find the output file.
        idx -- index of the file in the output file list.
        kind -- [None] the filter keyword.
            None -- don't apply the filter.
            'g' -- the good particles.
            'f' -- the fallback particles.
            'l' -- the particles loss along the beamline.
            'c' -- the particles loss at cathode.
            'b' -- the bad particles.

        Returns:
        data -- the information read.
        """
        try:
            fname = self._get_output_list(sim)[idx]
        except IndexError as err:
            raise AstraCoreError('output file at index {} not found!'.format(idx)) from err

        return self.get_data_by_name(sim, fname, kind)

    def _get_output_list(self, sim, full=False):
        """ Get the output files list in a given sim folder.

        Keyword arguments:
        sim -- simulation folder.
        full -- [False] if return the fullnames of the output files.

        Returns:
        out_fnames -- a list of the output filenames.
        """
        sim_path = os.path.join(self.root, sim)
        try:
            out_fnames = [f for f in os.listdir(sim_path) if self.is_output(f)]
            out_fnames.sort(key=lambda f: int(f.split('.')[1]))
            if full:
                out_fnames = [os.path.join(sim_path, f) for f in out_fnames]
        except FileNotFoundError as exc:
            raise AstraCoreError(
                'simulation folder {} not found!'.format(sim)) from exc

        return out_fnames

    def clear_log(self):
        """ Clear the log.
        """
        self.log = []

    def archive_log(self):
        """ Write the log into a text file.
        """
        with open(os.path.join(self.root, 'astracore.log'), 'wb') as f:
            pickle.dump(self.log, f)

        self.clear_log()

    def load_log(self):
        """ Load the log from the astracore.log file.
        """
        try:
            with open(os.path.join(self.root, 'astracore.log'), 'rb') as f:
                log = pickle.load(f)
                self.log = log + self.log
        except FileNotFoundError as exc:
            raise AstraCoreError(
                "astracore log file doesn't exist!") from exc

    def get_last_run_time(self):
        """ Get the running time of last simulation.

        Returns:
        run_time -- running time in second.
        """
        try:
            out = self.log[-1][3]
            run_time = self.get_run_time(out)
        except IndexError as exc:
            raise AstraCoreError("log is empty!") from exc

        return run_time

    def print_last_output(self):
        """ Print the astra output from last simulation.
        """
        try:
            out = self.log[-1][3]
        except IndexError as exc:
            raise AstraCoreError("log is empty!") from exc

        print(self.get_output(out))
