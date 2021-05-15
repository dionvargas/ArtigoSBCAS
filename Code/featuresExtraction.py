# ***************************************************************************************************************************
# Import libraries, modules, py files, etc
import glob
import sys

from bibs.dataManipulator import DataManipulator
from sfe.signal_transform import SignalTransform
from sfe.feature_extractor_1D import ODFeatureExtractor
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
path_dataset = "C:\\Users\\diona\\Dropbox\\Mestrado\\Tese\\Bases de Dados\\Universidade de Bonn\\"
path_data = "data.csv"

folders = ['Z', 'O', 'N', 'F', 'S']

delta_F = [1, 4]
theta_F = [4, 8]
alpha_F = [8, 12]
beta_F = [12, 30]
gamma_F = [30, 60]
total_F = [1, 60]

# ***************************************************************************************************************************
# Data Manipulator
dataManipulator = DataManipulator(path_data)
dataManipulator.remove_data()

# ***************************************************************************************************************************
# feature extraction
for i in range(0, len(folders)):
    file_path = path_dataset + folders[i] + "/*.txt"

    arq = glob.glob(file_path)

    for j in range(0, len(arq)):
        with open(arq[j], "r") as text_file:
            signal = text_file.readlines()
            signal = list(map(int, signal))

        print('Processando ' + arq[j])

        signal = np.asarray(signal)

        stObj = SignalTransform(signal, Fs=Fs)

        # ------------------------------------------------------------------------------
        # power spectrum generation
        ps = stObj.get_power_spectrum()
        f_grid = getfgrid(Fs, len(ps))

        # power spectrum of Delta Waves
        psDelta = spectrum_filter(ps, Fs, delta_F[0], delta_F[1])
        f_gridDelta = getfgrid(Fs, len(ps), fpassMin=delta_F[0], fpassMax=delta_F[1])

        # power spectrum feature extraction of Delta Waves
        psDeltaO = ODFeatureExtractor(psDelta, freq_grid=f_gridDelta[:-1], label='psDelta')
        psDeltaO.extract_all_features()
        psDeltaF = psDeltaO.get_extracted_features()

        # power spectrum of Theta Waves
        psTheta = spectrum_filter(ps, Fs, theta_F[0], theta_F[1])
        f_gridTheta = getfgrid(Fs, len(ps), fpassMin=theta_F[0], fpassMax=theta_F[1])

        # power spectrum feature extraction of Theta Waves
        psThetaO = ODFeatureExtractor(psTheta, freq_grid=f_gridTheta[:-1], label='psTheta')
        psThetaO.extract_all_features()
        psThetaF = psThetaO.get_extracted_features()

        # power spectrum of Alpha Waves
        psAlpha = spectrum_filter(ps, Fs, alpha_F[0], alpha_F[1])
        f_gridAlpha = getfgrid(Fs, len(ps), fpassMin=alpha_F[0], fpassMax=alpha_F[1])

        # power spectrum feature extraction of Alpha Waves
        psAlphaO = ODFeatureExtractor(psAlpha, freq_grid=f_gridAlpha[:-1], label='psAlpha')
        psAlphaO.extract_all_features()
        psAlphaF = psAlphaO.get_extracted_features()

        # power spectrum of Beta Waves
        psBeta = spectrum_filter(ps, Fs, beta_F[0], beta_F[1])
        f_gridBeta = getfgrid(Fs, len(ps), fpassMin=beta_F[0], fpassMax=beta_F[1])

        # power spectrum feature extraction of Beta Waves
        psBetaO = ODFeatureExtractor(psBeta, freq_grid=f_gridBeta[:-1], label='psBeta')
        psBetaO.extract_all_features()
        psBetaF = psBetaO.get_extracted_features()

        # power spectrum of Gamma Waves
        psGamma = spectrum_filter(ps, Fs, gamma_F[0], gamma_F[1])
        f_gridGamma = getfgrid(Fs, len(ps), fpassMin=gamma_F[0], fpassMax=gamma_F[1])

        # power spectrum feature extraction of Gamma Waves
        psGammaO = ODFeatureExtractor(psGamma, freq_grid=f_gridGamma[:-1], label='psGamma')
        psGammaO.extract_all_features()
        psGammaF = psGammaO.get_extracted_features()

        psAll = {**psDeltaF, **psThetaF, **psAlphaF, **psBetaF, **psGammaF}

        dataManipulator.save_data(psAll, i, titles=True)