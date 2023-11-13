import os
import shutil
import mne
import sys
import pyedflib


"""
Dependencies:
    1) mne 1.5.1
    2) Python 3.10+
    3) EDFlib-Python-1.0.8+

Usage:
    >>> python3 edf_merge.py <edf-file-path> <[optional] output-file-name>

Script Usage:
    1) Move script into a folder.
    2) Within that folder, create a folder named 'in'.
        - Move EDF files into 'in'.
    3) Execute script from folder directory, i.e.
        >>> python3 edf_merge.py
        - The default name of the output file is output.edf.
        - To change this, run the script with a command-line argument. Ex.
        >>> python3 edf_merge.py desired_name
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

    # Remove non scalp-eeg
    channels = data.ch_names
    scalp_start = channels.index('Fp1-Ref')
    to_drop = channels[:scalp_start]
    scalp_raw = data.drop_channels(to_drop)

    # Decimate 2000 hz to 200hz
    data = data.resample(200.0) # internally uses scipy.signal.decimate
    print_edf(data, 'Output')   
    return data

def concat(lst):
    """Concatenates and returns two mne.io.Raw objects"""
    return mne.concatenate_raws(lst)

def bipolar_reference(raw_edf):
    cathodes = ['Fp1-Ref', 'F7-Ref', 'T7-Ref', 'P7-Ref', 'Fp1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'Fz-Ref', 'Cz-Ref', 'Fp2-Ref', 'F4-Ref', 'C4-Ref', 'P4-Ref', 'Fp2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref']
    anodes =   ['F7-Ref' , 'T7-Ref', 'P7-Ref', 'O1-Ref', 'F3-Ref',  'C3-Ref', 'P3-Ref', 'O1-Ref', 'Cz-Ref', 'Pz-Ref', 'F4-Ref',  'C4-Ref', 'P4-Ref', 'O2-Ref', 'F8-Ref',  'T8-Ref', 'P8-Ref', 'O2-Ref']
    names =    ['Fp1_F7',  'F7_T7',  'T7_P7',  'P7_O1',  'Fp1_F3',  'F3_C3',  'C3_P3',  'P3_O1',  'Fz_Cz',  'Cz_Pz',  'Fp2_F4',  'F4_C4',   'C4_P4', 'P4_O2',  'Fp2_F8',  'F8_T8',  'T8_P8',  'P8_O2']
    return mne.set_bipolar_reference(raw_edf, anodes, cathodes, names)

def average_reference(raw_edf):
    return raw_edf.set_eeg_reference()

def export(raw_edf, target_name, overwrite_existing=True, bipolar=False, common_average=False, bipolar_common_average=False):
    """Export raw object as EDF file"""
    mne.export.export_raw(target_name + '.edf', raw_edf, overwrite=True)
    if bipolar:
        mne.export.export_raw(target_name + '-bipolar.edf', bipolar_reference(raw_edf), overwrite=overwrite_existing)
    if common_average:
        mne.export.export_raw(target_name + '-common-average.edf', average_reference(raw_edf), overwrite=overwrite_existing)
    if bipolar_common_average:
        mne.export.export_raw(target_name + '-bipolar-common-average.edf', average_reference(bipolar_reference(raw_edf)), overwrite=overwrite_existing)

def print_edf(raw_edf, name):
    """Print basic information about an mne.io.raw object."""
    print(f'\n\n\n\nTesting {name} EDF:\n')
    print(raw_edf.info)
    print('Dim:', raw_edf.get_data().shape[0], 'channels', 'x', raw_edf.get_data().shape[1], 'time points\n\n\n')

if __name__ == "__main__":
    # Set output file name to optional command-line argument, or 'output.edf' if no argument.
    if len(sys.argv) == 1:
        name = 'output'
    else:
        name = sys.argv[1]
    src = os.getcwd()
    sep = os.sep
    # Navigate to EDF files
    while os.getcwd() is not sep:
        os.chdir('..')
    edf_path_in = src + sep + 'in'
    os.chdir(edf_path_in)
    #for root, folders, files in os.walk(edf_path_in):

    #for root, folders, files in os.walk(edf_path_in):
        # print(root) - C:\Users\raymo\ray\urap\edf-merger\in
        # print(list(folders)) - []
        # print(list(files)) - ['PR03_0010.edf', 'PR03_0011.edf']
    src_files = list(sorted(os.listdir(edf_path_in)))
    merged = concat([trim_and_decim(edf, 200) for edf in src_files])
    os.chdir(src)
    #mne.export.export_raw('test.edf', merged)
    #export(merged, 'test')
    new = src + sep + 'out'
    os.mkdir(new)
    os.chdir(new)
    export(merged, name)