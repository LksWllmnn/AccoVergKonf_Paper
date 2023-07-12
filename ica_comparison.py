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

import csv
from time import time
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA


print(__doc__)

# %%
# Read and preprocess the data. Preprocessing consists of:
#
# - MEG channel selection
# - 1-30 Hz band-pass filter

# User_frame = pd.read_csv(
#     "./Nutzertests/09876_filtered.csv", sep=',', index_col=0)

# User_frame = User_frame.iloc[:-1]

# data = User_frame.to_numpy(dtype=np.float64)

# E: \Projekte\HFU\AccoVergKonf_Paper\Nutzertests\09876_Accel.csv


# Pfad zur CSV-Datei
# id = "09876"
# csv_file = f"E:/Projekte/HFU/AccoVergKonf_Paper/Nutzertests/{id}_filtered.csv"

# # Liste zum Speichern der korrigierten Zeilen
# corrected_lines = []

# # CSV-Datei öffnen und Zeilen überprüfen
# with open(csv_file, 'r') as file:
#     reader = csv.reader(file, delimiter=',')
#     print(reader)
#     for row in reader:
#         if len(row) == 29:
#             corrected_lines.append(row)
#         else:
#             # Korrektur der Zeile mit 30 Feldern oder Überspringen der Zeile
#             # Beispiel: Korrektur des letzten Feldes
#             row = row[:29]  # Nur die ersten 29 Felder behalten
#             corrected_lines.append(row)

# # Korrigierte Zeilen in eine neue CSV-Datei schreiben
# corrected_csv_file = f"E:/Projekte/HFU/AccoVergKonf_Paper/Nutzertests/{id}_corrected.csv"

# with open(corrected_csv_file, 'w', newline='') as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerows(corrected_lines)

# # DataFrame aus der korrigierten CSV-Datei erstellen
# User_frame = pd.read_csv(corrected_csv_file, index_col=0)

# Pfad zur CSV-Datei
csv_file = "./Nutzertests/09876_filtered.csv"

# DataFrame aus der CSV-Datei erstellen
df = pd.read_csv(csv_file, sep=',', index_col=0)

# Letzten Eintrag entfernen
df = df.iloc[:-1]

# Daten und Kanalnamen aus dem DataFrame extrahieren
data = df.to_numpy(dtype=float)
ch_names = df.columns.tolist()[:29]  # Nur die ersten 29 Kanalnamen extrahieren

# Kanaltypen festlegen
ch_types = ['eeg'] * 29

# Sampling-Frequenz festlegen
sfreq = 300

# MNE Info-Objekt erstellen
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# MNE RawArray erstellen
# Nur die ersten 29 Kanäle extrahieren
raw = mne.io.RawArray(data.T[:, :29], info)

# assigning the channel type when initializing the Info object
ch_names = ["Time", "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "CM", "A1", "Fp1", "Fp2", "T3", "T5", "O1", "O2", "X3",
            "X2", "F7", "F8", "X1", "A2", "T6", "T4", "Trigger", "Time_Offset", "ADC_Status", "ADC_Sequence", "Event"]


ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'ecg', 'misc', 'misc', 'misc']

sampling_freq = 300  # in Hertz

info = mne.create_info(ch_names=ch_names, sfreq=sampling_freq)
reject = dict(mag=5e-12, grad=4000e-13)
# raw = mne.io.RawArray(data, info)
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
