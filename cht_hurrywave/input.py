# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import datetime
import copy

class Variables:
    def __init__(self):
        pass

class HurryWaveInput:
    """
    A class to handle the input configuration for the HurryWave model. This class manages the 
    initialization, reading, writing, and updating of input parameters used in the model simulation.

    The class provides functionality to:

    - Initialize default values for various model parameters.
    - Read input values from an existing "hurrywave.inp" file and store them as model variables.
    - Write the current model configuration to the "hurrywave.inp" file.
    - Print the current configuration for inspection.
    - Update specific parameters in the model configuration.

    Key Methods:

    - __init__:
         Initializes the input parameters with default values.
    - read:
         Reads the "hurrywave.inp" file and updates the model parameters.
    - write: 
         Writes the current input parameters to the "hurrywave.inp" file.
    - print:
         Prints the current configuration of the model's input parameters to the console.
    - update:
         Updates the model's input parameters with new values.

    Arguments:
    - hw: The HurryWave model object to which the input parameters belong.
    """
    def __init__(self, hw):

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
        self.variables.redopt = 1
        self.variables.winddrag = "zijlema"
        self.variables.cdcap = 0.0025

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

        # self.variables.cdnrb = 3
        # self.variables.cdwnd = [0.0, 28.0, 50.0]
        # self.variables.cdval = [0.001, 0.0025, 0.0015]

    def read(self):
        # Reads hurrywave.inp
        file_name = os.path.join(self.model.path, "hurrywave.inp")
        fid = open(file_name, 'r')
        lines = fid.readlines()
        fid.close()
        for line in lines:
            str = line.split("=")
            if len(str) == 1:
                # Empty line
                continue
            name = str[0].strip()
            val = str[1].strip()
            try:
                # First try to convert to int
                val = int(val)
            except ValueError:
                try:
                    # Now try to convert to float
                    val = float(val)
                except:
                    pass
            if name == "tref":
                val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
            if name == "tstart":
                val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
            if name == "tstop":
                val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
            setattr(self.variables, name, val)

    def write(self):

        file_name = os.path.join(self.model.path, "hurrywave.inp")
        variables = copy.copy(self.variables)
        # Remove some input variables
        if self.model.boundary_conditions.forcing == "spectra":
            variables.bhsfile = None
            variables.btpfile = None
            variables.bwdfile = None
            variables.bdsfile = None
        else:
            variables.bspfile = None

        fid = open(file_name, "w")
        for key, value in variables.__dict__.items():
            if not value is None:
                if type(value) == "float":
                    string = f'{key.ljust(20)} = {float(value)}\n'
                elif type(value) == "int":
                    string = f'{key.ljust(20)} = {int(value)}\n'
                elif type(value) == list:
                    valstr = ""
                    for v in value:
                        valstr += str(v) + " "
                    string = f'{key.ljust(20)} = {valstr}\n'
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f'{key.ljust(20)} = {dstr}\n'
                else:
                    string = f'{key.ljust(20)} = {value}\n'
                fid.write(string)
        fid.close()

    def print(self):
        for key, value in self.variables.__dict__.items():
            if not value is None:
                if type(value) == "float":
                    string = f'{key.ljust(20)} = {float(value)}\n'
                elif type(value) == "int":
                    string = f'{key.ljust(20)} = {int(value)}\n'
                elif type(value) == list:
                    valstr = ""
                    for v in value:
                        valstr += str(v) + " "
                    string = f'{key.ljust(20)} = {valstr}\n'
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f'{key.ljust(20)} = {dstr}\n'
                else:
                    string = f'{key.ljust(20)} = {value}\n'
                print(string)

    def update(self, pars):
        for key in pars:
            setattr(self.variables, key, pars[key])
