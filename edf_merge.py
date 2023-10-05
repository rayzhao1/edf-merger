# python3 -m pip install numpy
import numpy as np
import os
import shutil
import scipy

import pyedflib
from pyedflib import highlevel
import mne
# common average referencing
"""
Script Usage:
    1) Move script into a folder.
    2) Within that folder, create a folder named 'in'.
        - Move EDF files into 'in'.
    3) Execute script from folder directory, i.e.
        > python3 edf_merge.py
"""

def trim_and_decim(edf_path, freq):
    """Takes a EDF file path and returns the EDF data as an mne.io.Raw object with:
        - Only scalp channels included.
        - Resamples input EDF's frequency to 'freq'.
    """
    data = mne.io.read_raw_edf(edf_path) # alt preload=True, data.get_data() # alt *._data

    # Splice away 'POL'
    dict = {name: name[4:] for name in data.ch_names}
    data = data.rename_channels(dict)

    # remove non scalp-eeg
    channels = data.ch_names
    scalp_start = channels.index('Fp1-Ref')
    to_drop = channels[:scalp_start]
    scalp_raw = data.drop_channels(to_drop)

    # Decimate 2000 hz to 200hz
    data = data.resample(200.0) # internally uses scipy.signal.decimate
    print_edf(data, 'Output')   
    return data

def concat(raw_a, raw_b):
    """Concatenates and returns two mne.io.Raw objects"""
    return mne.concatenate_raws([raw_a, raw_b])

def bipolar_reference(raw_edf):
    channels = raw_edf.ch_names
    cathodes = ['Fp1-Ref', 'F7-Ref', 'T7-Ref', 'P7-Ref', 'Fp1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'Fz-Ref', 'Cz-Ref', 'Fp2-Ref', 'F4-Ref', 'C4-Ref', 'P4-Ref', 'Fp2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref']
    anodes =   ['F7-Ref' , 'T7-Ref', 'P7-Ref', 'O1-Ref', 'F3-Ref',  'C3-Ref', 'P3-Ref', 'O1-Ref', 'Cz-Ref', 'Pz-Ref', 'F4-Ref',  'C4-Ref', 'P4-Ref', 'O2-Ref', 'F8-Ref',  'T8-Ref', 'P8-Ref', 'O2-Ref']
    names =    ['Fp1_F7',  'F7_T7',  'T7_P7',  'P7_O1',  'Fp1_F3',  'F3_C3',  'C3_P3',  'P3_O1',  'Fz_Cz',  'Cz_Pz',  'Fp2_F4',  'F4_C4',   'C4_P4', 'P4_O2',  'Fp2_F8',  'F8_T8',  'T8_P8',  'P8_O2']
    return mne.set_bipolar_reference(raw_edf, anodes, cathodes, drop_refs = True)

def average_reference(raw_edf):
    return raw_edf.set_eeg_reference()

def export(raw_edf, target_name, bipolar=True, common_average=True, bipolar_common_average=True):
    """Export raw object as EDF file"""
    mne.export.export_raw(target_name+'.edf', raw_edf, overwrite=True)
    if bipolar:
        mne.export.export_raw(target_name + '-bipolar.edf', bipolar_reference(raw_edf), overwrite=True)
    if common_average:
        mne.export.export_raw(target_name + '-common-average.edf', average_reference(raw_edf), overwrite=True)
    if bipolar_common_average:
        mne.export.export_raw(target_name + '-bipolar-common-average.edf', average_reference(bipolar_reference(raw_edf)), overwrite=True)

def print_edf(raw_edf, name):
    """Print basic information about an mne.io.raw object."""
    print(f'\n\n\n\nTesting {name} EDF:\n')
    print(raw_edf.info)
    print('Dim:', raw_edf.get_data().shape[0], 'channels', 'x', raw_edf.get_data().shape[1], 'time points\n\n\n')

if __name__ == "__main__":
    src = os.getcwd()
    edf_path_in = src + '\\in'
    os.chdir(edf_path_in)

    """
    for root, folders, files in os.walk(edf_path_in):
        for folder in folders:
            os.chdir(folder)
            path = os.path.join
            lst_of_raws = list(map(lambda r: mne.io.read_raw_edf(edf_path)))
            merged = mne.concatenate_raws(lst_of_raws)

    """

    src_files = list(sorted(os.listdir(edf_path_in)))
    a, b = src_files[0], src_files[1]
    a, b = trim_and_decim(a, 200), trim_and_decim(b, 200)

    merged = concat(a, b)
    print_edf(merged, 'Merged')

    # export
    export(merged, 'test')