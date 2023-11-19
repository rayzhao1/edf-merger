import os
import shutil
import mne
import sys
import csv
import datetime

"""
Dependencies:
    1) mne 1.5.1
    2) Python 3.10+
    3) EDFlib-Python-1.0.8+

Usage:
    python3 edf_merge.py <edf-file-path> <[optional] output-file-name> <[Merge-checking (optional)] False>

    - "Merge-checking" is on by default and will throw an error if the script attempts to merge EDF files that are not time-continuous. 
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

def get_first_date(csv_in):
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _, first_row = next(csv_reader), next(csv_reader)
        time_str, time_format = first_row[3], '%Y-%m-%d %H:%M:%S.%f'
        return datetime.datetime.strptime(time_str, time_format)  

def parse_find(csv_in, start, end, detect=True):
    """Iterate through 'csv_in' and return the name of the .edf file with 'edf_start' closest to start and 'edf_end' closest to end.
        - Returns these two respective files as a two-element tuple.
        - If detect=True, also return a list of files that differ in time from their previous file by an amount >= margin (1.5 min by default).
    
    """
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    files = []
    curr = start
    flag = False
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            curr = datetime.datetime.strptime(row[3], time_format) 
            if curr > start:
                flag = True
            if not flag:
                continue
            if curr > end:
                break
            files.append(row[1])
    return files

if __name__ == "__main__": # can get rid of
    # Process command-line args
    argc = len(sys.argv)
    assert argc in range(2, 4), "Please run the script in the following format: python3 edf_merge.py <edf-file-path> <[optional] output-file-name>"
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

    # Merge EDF files
    patient = os.path.basename(os.getcwd())
    csv_meta = f'{patient}_EDFMeta.csv'
    print(csv_meta)
    assert csv_meta in os.listdir(), f'Please provide a .csv file formatted as: {patient}_EDFMeta.csv'

    # 2021-05-25 17:49:59.920000
    # Identify start and end based on .csv, PR03_EDFMeta.csv
    t0 = get_first_date(csv_meta)
    t0.replace(hour = 9, minute = 0, second = 0, microsecond = 0)
    duration = datetime.timedelta(hours = 11)
    tf = t0 + duration
    edf_files = parse_find(csv_meta, t0, tf)
    merged = concat([trim_and_decim(edf, 200) for edf in edf_files])
    os.chdir(src)
    os.mkdir(f'out-{patient}')
    export(merged, name)
    """
    os.chdir(patient)
    edf_files = os.listdir().sort()
    if not discontinuous:
        start, end = edf_files.index(start_file), edf_files.index(end_file)
        edf_files = edf_files[start:end]
        merged = concat([trim_and_decim(edf, 200) for edf in edf_files if edf[-4:] == '.edf'])

    # Make folder output folder
    os.chdir(src)
    new = src + sep + 'out'
    os.mkdir(new)
    # Export to output folder
    os.chdir(new)
    export(merged, name)
    """

    # Unchecked concatenation
    # merged = concat([trim_and_decim(edf, 200) for edf in edf_files])

    """
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


    """

Extract start date, (datetime, or unix)

set file duration (ex. 11 hours)

inputs:
- output file duration
- number of nights (default=run to end)

for every row

return [(start1, end1), (start2, end2), ...]

test = datetime.datetime(2019, 1, 1)

=(D2-DATE(1970,1,1)) * 86400

>>> test = datetime.datetime(2019, 1, 1)
>>> test
datetime.datetime(2019, 1, 1, 0, 0)
>>> print(test)
2019-01-01 00:00:00
>>> time = test + timedate.timedelta(hours = 20)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'timedate' is not defined
>>> time = test + datetime.timedelta(hours = 20)
>>> print(time)
2019-01-01 20:00:00
>>> s = 1621964962
>>> print(s)
1621964962
>>> datetime.datetime.utcfromtimestamp(s)
datetime.datetime(2021, 5, 25, 17, 49, 22)
>>> a = datetime.datetime.utcfromtimestamp(s)
>>> datetime.datetime(a.date)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'builtin_function_or_method' object cannot be interpreted as an integer
>>> a.date
<built-in method date of datetime.datetime object at 0x000001A500E5BAE0>
>>> print(a.date)
<built-in method date of datetime.datetime object at 0x000001A500E5BAE0>
>>> a.date()
datetime.date(2021, 5, 25)

        #csv_reader = csv.reader(csv_file, delimiter=',')
        #time_str, time_format = next(csv_reader)[3], '%Y-%m-%d %H:%M:%S.%f'
        #last = datetime.datetime.strptime(time_str, time_format)
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)    # remove header row
        for row in csv_reader:
            # If located both indexes, break.
            if flag == 2 and not detect:
                break
            time_str, time_format = row[3], '%Y-%m-%d %H:%M:%S.%f'
            curr = datetime.datetime.strptime(time_str, time_format)    
            # Detect discontinuous rows  
    
            #if detect and curr - last > margin:
            #    discontinuous.append(row[0]) 
            #    flag += 1    
          
            # Look for start_file 
            print(type(curr - target))
            if flag == 0 and curr >= target: # 11:59 - 12:00, 12:01 - 12:00
                start_file = row[1]
                target = end
                flag += 1
            # Look for end_index
            if flag == 1 and curr >= target:
                end_file = row[1]
                flag += 1
            # last = curr
        return (start_file, end_file, discontinuous) if detect else (start_file, end_file)

datetime object
parsed_datetime = datetime.strptime(your_date_string, date_format)
"""

