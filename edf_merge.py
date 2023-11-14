import os
import shutil
import mne
import sys
import csv

"""
Dependencies:
    1) mne 1.5.1
    2) Python 3.10+
    3) EDFlib-Python-1.0.8+

Usage:
    python3 edf_merge.py <edf-file-path> <[optional] output-file-name> <[Merge-checking (optional)] False>

    - "Merge-checking" is on by default and will throw an error if the script attempts to merge EDF files that are not time-contiguous. 
    - However, it is less performant. It can be turned off with an additional argument, False.
    - The output file is created in the current working directory, in a folder titled 'out'.

Requirements:
    1) The <edf-file-path> directory stores the EDF files to be merged in a folder with the same name.
    2) If "Merge-checking" is on, the <edf-file-path> directory contains a '<name>_EDFMeta.csv'.

Example - Valid File Structure:
    .
    └── data_store0
        └── presidio
            └── nihon_kohden
                └── PR05
                    └── PR05
                        ├── PR05_EDFMeta.csv
                        └── PR05
                            ├── PR05_2605.edf
                            ├── PR05_2606.edf
                            └── PR05_2607.edf
                
    python3 edf_merge.py data_store0/presidio/nihon_kohden/PR05 False
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

def concat(a, b):
    """Concatenates two mne.io.Raw objects and returns result."""
    return mne.concatenate_raws(lst)

def concat(lst):
    """Concatenates a list of mne.io.Raw objects and returns result."""
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
    # Process command-line args
    argc = len(sys.argv)
    if argc not in range(2, 4):
        raise Exception("Please run the script in the following format: python3 edf_merge.py <edf-file-path> <[optional] output-file-name>")
    patient_path = sys.argv[1]
    if argc == 3:
        name = sys.argv[2]
    else:
        name = 'output'
    src = os.getcwd()
    sep = os.sep

    # Navigate to EDF files
    while os.getcwd() is not sep:
        os.chdir('..')
    os.chdir(patient_path)  

    csv_name = f'{edf_file}'

    # Merge EDF files
    patient = os.path.basename(os.getcwd())
    if f'{patient}_EDFMeta.csv' not in os.listdir():
        raise Exception("Please provide a .csv file formatted as: <patient>_EDFMeta.csv")

    # Unchecked concatenation
    # merged = concat([trim_and_decim(edf, 200) for edf in edf_files])

    # Identify start and end based on .csv, PR03_EDFMeta.csv
    with open(f'{patient}_EDFMeta.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        start = 0
        end = 0
        
    os.chdir(patient)
    edf_files = list(sorted(os.listdir()))

    while int(next(edf_files)[-2:]) != start:
        pass
    merged = edf_files[0]
    last = merged[-2:]
    blank = None
    start, end = 0, 0
    for n, edf_file in zip(range(start, end), edf_files): # 11 hrs * 12 recordings/hr = max of 132 files)
        if int(last) + 1 = int(edf_file[-2:]):
            merged = concat(merged, edf)
        else:
            merged = concat(merged, blank)

    """
    # Make folder output folder
    os.chdir(src)
    new = src + sep + 'out'
    os.mkdir(new)
    # Export to output folder
    os.chdir(new)
    export(merged, name)"""