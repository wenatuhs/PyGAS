import os
import numpy as np


class LTEParser:
    """ A tool that eases the painful scans in ELEGANT simulations.
    """

    def __init__(self, root):
        """ Initialize the Elegant commander by passing a simulation directory to it.

        Keyword arguments:
        root -- the path to the root simulation folder.
        """
        self.log = []
        self.root = root
        self.lattice = None
        self.main = None
        self._get_project_info()

    def read(self, fname=None):
        """ Peek into the content of the input file.

        Keyword arguments:
        fname -- [None] filename of the Elegane input file with extension.
            if None, peek into the self.main file.

        Returns:
        nml -- a f90 namelist contains all the infomations.
        """
        if fname is None:
            fname = self.main
        fullname = os.path.join(self.root, fname)
        nml = lp.read(fullname)

        return nml
