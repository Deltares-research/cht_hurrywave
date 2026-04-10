"""
HurryWave input file handling.

Provides :class:`HurryWaveInput`, which reads and writes the ``hurrywave.inp``
key-value parameter file and stores all model configuration variables.
"""

import copy
import datetime
import os


class Variables:
    """Container for HurryWave input variables."""

    def __init__(self) -> None:
        pass


class HurryWaveInput:
    """
    Manage the ``hurrywave.inp`` configuration file.

    Initialises default values for all model parameters, and provides
    methods to read from and write to disk.

    Parameters
    ----------
    hw : HurryWave
        Parent HurryWave model instance.
    """

    def __init__(self, hw) -> None:
        self.model = hw

        now = datetime.datetime.now()
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)

        self.variables = Variables()
        self.variables.mmax = 0
        self.variables.nmax = 0
        self.variables.dx = 0.1
        self.variables.dy = 0.1
        self.variables.x0 = 0.0
        self.variables.y0 = 0.0
        self.variables.rotation = 0.0
        self.variables.latitude = 0.0
        self.variables.tref = now
        self.variables.tstart = now
        self.variables.tstop = now + datetime.timedelta(days=2)
        self.variables.dt = 300.0
        self.variables.tspinup = 7200.0
        self.variables.t0out = -999.0
        self.variables.dtmapout = 3600.0
        self.variables.dthisout = 600.0
        self.variables.dtrstout = 0.0
        self.variables.dtsp2out = 3600.0
        self.variables.dtmaxout = 0.0
        self.variables.trstout = -999.0
        self.variables.dtwnd = 1800.0
        self.variables.rhoa = 1.25
        self.variables.rhow = 1024.0
        self.variables.dmx1 = 0.2
        self.variables.dmx2 = 0.00001
        self.variables.crsgeo = 0
        self.variables.freqmin = 0.04
        self.variables.freqmax = 0.5
        self.variables.nsigma = 12
        self.variables.ntheta = 36
        self.variables.crs_name = "WGS 84"
        self.variables.crs_type = "geographic"
        self.variables.crs_utmzone = None
        self.variables.crs_epsg = None
        self.variables.gammajsp = 3.3
        self.variables.spinup_meteo = 1
        self.variables.quadruplets = 1
        self.variables.redopt = 1
        self.variables.winddrag = "zijlema"
        self.variables.cdcap = 0.0025

        self.variables.qtrfile = None
        self.variables.depfile = None
        self.variables.mskfile = None
        self.variables.bndfile = None
        self.variables.bhsfile = None
        self.variables.btpfile = None
        self.variables.bwdfile = None
        self.variables.bdsfile = None
        self.variables.bspfile = None
        self.variables.rstfile = None
        self.variables.spwfile = None
        self.variables.amufile = None
        self.variables.amvfile = None
        self.variables.wndfile = None
        self.variables.obsfile = None
        self.variables.ospfile = None
        self.variables.wblfile = None

        self.variables.inputformat = "bin"
        self.variables.outputformat = "net"

    def read(self) -> None:
        """
        Read ``hurrywave.inp`` from disk and populate :attr:`variables`.

        Raises
        ------
        FileNotFoundError
            If ``hurrywave.inp`` does not exist in the model path.
        """
        file_name = os.path.join(self.model.path, "hurrywave.inp")
        with open(file_name, "r") as fid:
            lines = fid.readlines()
        for line in lines:
            parts = line.split("=")
            if len(parts) == 1:
                continue
            name = parts[0].strip()
            val = parts[1].strip()
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except Exception:
                    pass
            if name == "tref":
                val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
            if name == "tstart":
                val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
            if name == "tstop":
                val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
            setattr(self.variables, name, val)

    def write(self) -> None:
        """
        Write current :attr:`variables` to ``hurrywave.inp``.

        Grid geometry variables are omitted when a quadtree file is active.
        Boundary spectra or time-series variables are suppressed depending on
        the active forcing type.
        """
        file_name = os.path.join(self.model.path, "hurrywave.inp")
        variables = copy.copy(self.variables)

        if self.model.input.variables.qtrfile is not None:
            variables.x0 = None
            variables.y0 = None
            variables.dx = None
            variables.dy = None
            variables.mmax = None
            variables.nmax = None
            variables.rotation = None

        if self.model.boundary_conditions.forcing == "spectra":
            variables.bhsfile = None
            variables.btpfile = None
            variables.bwdfile = None
            variables.bdsfile = None
        else:
            variables.bspfile = None

        with open(file_name, "w") as fid:
            for key, value in variables.__dict__.items():
                if value is not None:
                    if type(value) == "float":
                        string = f"{key.ljust(20)} = {float(value)}\n"
                    elif type(value) == "int":
                        string = f"{key.ljust(20)} = {int(value)}\n"
                    elif type(value) == list:
                        valstr = " ".join(str(v) for v in value)
                        string = f"{key.ljust(20)} = {valstr}\n"
                    elif isinstance(value, datetime.date):
                        dstr = value.strftime("%Y%m%d %H%M%S")
                        string = f"{key.ljust(20)} = {dstr}\n"
                    else:
                        string = f"{key.ljust(20)} = {value}\n"
                    fid.write(string)

    def print(self) -> None:
        """Print all non-None variables to stdout in ``key = value`` format."""
        for key, value in self.variables.__dict__.items():
            if value is not None:
                if type(value) == "float":
                    string = f"{key.ljust(20)} = {float(value)}\n"
                elif type(value) == "int":
                    string = f"{key.ljust(20)} = {int(value)}\n"
                elif type(value) == list:
                    valstr = " ".join(str(v) for v in value)
                    string = f"{key.ljust(20)} = {valstr}\n"
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f"{key.ljust(20)} = {dstr}\n"
                else:
                    string = f"{key.ljust(20)} = {value}\n"
                print(string)

    def update(self, pars: dict) -> None:
        """
        Update one or more input variables.

        Parameters
        ----------
        pars : dict
            Mapping of variable name to new value.
        """
        for key in pars:
            setattr(self.variables, key, pars[key])
