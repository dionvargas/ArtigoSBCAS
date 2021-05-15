# ***************************************************************************************************************************
# Import libraries, modules, py files, etc
import glob
import sys
import os.path
import matplotlib.pyplot as plt

from sfe.signal_transform import SignalTransform
from sfe.sfe_aux_tools import getfgrid, spectrum_filter

import numpy as np

if not ('numpy' in sys.modules):
    pass

if not ('glob' in sys.modules):
    import glob as glob

# ***************************************************************************************************************************
# global variables
Fs = 173.61  # frequency sample

# path of Bonn EEG dataset
folders = []
folders.append("..\\Bases de Dados\\Universidade de Bonn\\Z\\Z001.txt")
folders.append("..\\Bases de Dados\\Universidade de Bonn\\N\\N001.txt")
folders.append("..\\Bases de Dados\\Universidade de Bonn\\S\\S001.txt")

labels = []
labels.append('A')
labels.append('B')
labels.append('C')
labels.append('D')
labels.append('E')

signals = []
times = []
powerSpectrun = []
f_grids = []

for f in folders:
    if(os.path.isfile(f)):
        with open(f, "r") as text_file:
            signal = text_file.readlines()
            signal = list(map(int, signal))
            signals.append(signal)
            time = np.zeros(len(signal))
            for i in range(len(time)):
                time[i] = i/173.61
            times.append(time)
    else:
        print(f + " não encontrado.")

for s in signals:
    signal = np.asarray(s)

    stObj = SignalTransform(signal, Fs=Fs)

    # ------------------------------------------------------------------------------
    # power spectrum generation
    ps = stObj.get_power_spectrum()
    powerSpectrun.append(10*np.log10(ps))
    f_grid = getfgrid(Fs, len(ps))/2
    f_grids.append(f_grid)

fig = plt.figure()
gs = fig.add_gridspec(len(signals), hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Sinais')
for i in range(len(signals)):
    axs[i].plot(times[i], signals[i])
    #axs[i].legend(labels[i], loc='upper center')

plt.show()

fig = plt.figure()
gs = fig.add_gridspec(len(signals), hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Espectro de Potência')
for i in range(len(powerSpectrun)):
    axs[i].plot(f_grids[i], powerSpectrun[i])
    #axs[i].legend(labels[i], loc='upper center')

plt.show()