"""
.. _ex-ica-comp:

===========================================
Compare the different ICA algorithms in MNE
===========================================

Different ICA algorithms are fit to raw MEG data, and the corresponding maps
are displayed.

"""
# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: BSD-3-Clause

# %%

from time import time
import numpy as np
import pandas as pd 
import mne
from mne.preprocessing import ICA
from mne.datasets import sample


print(__doc__)

# %%
# Read and preprocess the data. Preprocessing consists of:
#
# - MEG channel selection
# - 1-30 Hz band-pass filter

User_frame = pd.read_csv("C:/Users/nicka/Desktop/Nutzertests/50944_filtered.csv", sep=',', index_col=0) 

data= pd.DataFrame.to_numpy(User_frame, dtype= np.float64)

# assigning the channel type when initializing the Info object
ch_names = ["Time","P3","C3","F3","Fz","F4","C4","P4","Cz","CM","A1","Fp1","Fp2","T3","T5","O1","O2","X3","X2","F7","F8","X1","A2","T6","T4","Trigger","Time_Offset","ADC_Status","ADC_Sequence","Event","Comments"]



ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 
            'eeg', 'eeg', 'eeg', 'ecg', 'misc', 'misc', 'misc']

sampling_freq = 300  # in Hertz

info = mne.create_info(ch_names= ch_names, sfreq= sampling_freq)
reject = dict(mag=5e-12, grad=4000e-13)
raw = mne.io.RawArray(data, info)
raw.filter(1, 30, fir_design="firwin")

# %%
# Define a function that runs ICA on the raw MEG data and plots the components


def run_ica(method, fit_params=None):
    ica = ICA(
        n_components=20,
        method=method,
        fit_params=fit_params,
        max_iter="auto",
        random_state=0,
    )
    t0 = time()
    ica.fit(raw, reject=reject)
    fit_time = time() - t0
    title = "ICA decomposition using %s (took %.1fs)" % (method, fit_time)
    ica.plot_components(title=title)


# %%
# FastICA
run_ica("fastica")

# %%
# Picard
run_ica("picard")

# %%
# Infomax
run_ica("infomax")

# %%
# Extended Infomax
run_ica("infomax", fit_params=dict(extended=True))
