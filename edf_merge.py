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
    python3 edf_merge.py <edf-file-path> <[optional] output-file-name> <[optional] start-time> <[optional] duration-time>

Requirements:
    1) The <edf-file-path> directory stores the EDF files to be merged in a folder with the same name.
    2) If "Merge-checking" is on, the <edf-file-path> directory contains a '<name>_EDFMeta.csv'.

Example - Valid File Structure:
    .
    └── data_store0
        └── presidio
            └── nihon_kohden
                └── PR05
                    ├── PR05_EDFMeta.csv
                    └── PR05
                        ├── PR05_2605.edf
                        ├── PR05_2606.edf
                        └── PR05_2607.edf
                
    python3 edf_merge.py data_store0/presidio/nihon_kohden/PR05 False
"""


class Interval:
    def __init__(self, files=[]):
        self.files = files
        self.start = None
        self.end = None
        self.pos = 0

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos == len(self.files):
            return StopIteration
        val = self.files[self.pos]
        self.pos += 1
        return val

    def add(self, file):
        self.files.append(file)


def to_edf(edf_path: str):
    source_path = os.getcwd()
    os.chdir(EDFS_PATH)
    raw_edf = mne.io.read_raw_edf(edf_path)
    os.chdir(source_path)
    return raw_edf


def trim_and_decimate(raw_edf: mne.io.Raw, freq: int) -> mne.io.Raw:
    """Takes an EDF file path and returns the EDF data as an mne.io.Raw object with:
        - Only scalp channels included.
        - Resamples input EDF's frequency to 'freq'.
    """

    print_edf(raw_edf, 'before')
    # Splice away 'POL'
    rename_dict: dict[str: str] = {name: name[4:] for name in raw_edf.ch_names}
    raw_edf = raw_edf.rename_channels(rename_dict)
    # Remove non scalp-eeg
    channels: list[str] = raw_edf.ch_names
    scalp_start: int = channels.index('Fp1-Ref')
    to_drop = channels[:scalp_start]
    raw_scalp = raw_edf.drop_channels(to_drop)
    # Decimate 2000 hz to 200hz
    raw_scalp = raw_scalp.resample(freq)  # internally uses scipy.signal.decimate
    print_edf(raw_scalp, 'Output')
    return raw_scalp


def concatenate(lst: list[mne.io.Raw]) -> mne.io.Raw:
    """Concatenates a list of mne.io.Raw objects and returns result."""

    return mne.concatenate_raws(lst)

def concatenate(a: mne.io.Raw, b: mne.io.Raw) -> mne.io.Raw:
    """Concatenates a list of mne.io.Raw objects and returns result."""

    return mne.concatenate_raws([a, b])

def bipolar_reference(raw_edf: mne.io.Raw) -> mne.io.Raw:
    cathodes = ['Fp1-Ref', 'F7-Ref', 'T7-Ref', 'P7-Ref', 'Fp1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'Fz-Ref', 'Cz-Ref',
                'Fp2-Ref', 'F4-Ref', 'C4-Ref', 'P4-Ref', 'Fp2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref']
    anodes = ['F7-Ref', 'T7-Ref', 'P7-Ref', 'O1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'O1-Ref', 'Cz-Ref', 'Pz-Ref',
              'F4-Ref', 'C4-Ref', 'P4-Ref', 'O2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref', 'O2-Ref']
    names = ['Fp1_F7', 'F7_T7', 'T7_P7', 'P7_O1', 'Fp1_F3', 'F3_C3', 'C3_P3', 'P3_O1', 'Fz_Cz', 'Cz_Pz', 'Fp2_F4',
             'F4_C4', 'C4_P4', 'P4_O2', 'Fp2_F8', 'F8_T8', 'T8_P8', 'P8_O2']
    return mne.set_bipolar_reference(raw_edf, anodes, cathodes, names)


def average_reference(raw_edf: mne.io.Raw) -> mne.io.Raw:
    return raw_edf.set_eeg_reference()


def export(raw_edf: mne.io.Raw, target_name: str, mode: str, overwrite_existing=True):
    """Export raw object as EDF file"""
    name: str = f'{target_name}.edf'
    match mode:
        case 'bipolar':
            mne.export.export_raw(name, bipolar_reference(raw_edf), 'edf', overwrite=overwrite_existing)
        case 'common_average':
            mne.export.export_raw(name, average_reference(raw_edf), 'edf', overwrite=overwrite_existing)
        case 'bipolar_common_average':
            mne.export.export_raw(name, average_reference(bipolar_reference(raw_edf)), 'edf', overwrite=overwrite_existing)
        case _: # default
            mne.export.export_raw(name, raw_edf, 'edf', overwrite=overwrite_existing)


def print_edf(raw_edf: mne.io.Raw, name: str):
    """Print basic information about an mne.io.Raw object."""
    data, time = raw_edf[:, :]
    print(f'\n\n\n\nTesting {name} EDF:\n')
    print(raw_edf.info)
    print('Dim:', raw_edf.get_data().shape[0], 'channels', 'x', raw_edf.get_data().shape[1], 'time points\n\n\n')


def write_txt(*args):
    with open(os.path.join(os.getcwd(), 'summary.txt'), 'a') as f:
        for txt in args:
            f.write(txt + '\n')
        f.write('\n\n\n')


def str_to_time(time_str: str, time_format='%Y-%m-%d %H:%M:%S.%f'):
    return datetime.datetime.strptime(time_str, time_format)


def get_first_date(csv_in: str):
    with open(csv_in) as csv_file:
        csv_reader: csv.reader = csv.reader(csv_file, delimiter=',')
        _, first_row = next(csv_reader), next(csv_reader)
        return str_to_time(first_row[3])


def parse_find(csv_in: str, start: datetime.datetime, end: datetime.datetime, all_files: set[str], margin=datetime.timedelta(seconds=15), limit: float = float('inf')) -> list[Interval]:
    """Iterate through 'csv_in' and return a list of lists, where each sublist contains an interval of EDF file names
       such that each EDF is less than 'margin' away from the previous file in time. This implementation relies on the
       fact that csv_in is sorted in time-chronological order. All returned EDF files are also constrained to be in the
       time range between 'start' and 'end'.
    """

    source_path: str = os.getcwd()
    os.chdir(PATIENT_PATH)
    time_format: str = '%Y-%m-%d %H:%M:%S.%f'
    files: list[Interval] = []
    count: int = 0
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # remove header

        row_0: list[str] = next(csv_reader)
        prev_name: str = row_0[1]
        prev_time_end: datetime.datetime = str_to_time(row_0[4])
        files.append(Interval([prev_name]))

        for row in csv_reader:
            curr_name: str = row[1]
            curr_time_start: datetime.datetime = str_to_time(row[3])
            # Do not record files earlier than start.
            if curr_time_start < start:
                continue
            # Done recording once endtime has been exceeded.
            if curr_time_start > end or count >= limit:
                break
            # If the time difference is large, add a new list subsection.
            if curr_time_start - prev_time_end > margin:
                files.append(Interval())
            # Add to list subsection, if the file exists.
            if curr_name in all_files:
                files[-1].add(curr_name)
                count += 1
            prev_time_end: datetime.datetime = datetime.datetime.strptime(row[4], time_format)

    os.chdir(source_path)
    return files


if __name__ == "__main__":  # can get rid of
    # Process command-line args
    argc: int = len(sys.argv)
    limit: float = float('inf')
    name, tag = 'output', ''
    assert argc in range(2, 5), "Incorrect script parameters. Use {...}"
    if argc == 4:
        tag = sys.argv[2]
        limit = float(sys.argv[3])
    elif argc == 3 and sys.argv[2].isnumeric():
        limit = float(sys.argv[2])
    elif argc == 3:
        tag = sys.argv[2]

    if tag:
        tag = '-' + tag

    SRC_PATH: str = os.getcwd()
    # Navigate to EDF files
    while os.getcwd() is not os.sep: # for server: for _ in range(1):
        os.chdir('..')
    HOME_PATH: str = os.getcwd()
    PATIENT_PATH: str = os.path.join(HOME_PATH, sys.argv[1])
    os.chdir(PATIENT_PATH)

    # Identify file paths
    PATIENT: str = os.path.basename(os.getcwd())
    EDFS_PATH: str = os.path.join(HOME_PATH, PATIENT_PATH, PATIENT)
    csv_meta: str = f'{PATIENT}_EDFMeta.csv'
    assert csv_meta in os.listdir(), f'Please provide a .csv file formatted as: {PATIENT}_EDFMeta.csv'

    # Identify start and end based on .csv, PR03_EDFMeta.csv
    t0: datetime.datetime = get_first_date(csv_meta)  # Set start date
    t0 = t0.replace(hour=21, minute=0, second=0, microsecond=0)  # Set start date and time
    duration: datetime.timedelta = datetime.timedelta(hours=11)  # Set target duration
    tf: datetime.datetime = t0 + duration  # Set end time

    # Retrieve list of sub lists. Each sublist is a set of continuous, in-range file names.
    os.chdir(EDFS_PATH)
    all_edfs: set[str] = set(os.listdir())
    os.chdir(PATIENT_PATH)
    edf_files: list[Interval] = parse_find(csv_meta, t0, tf, all_edfs, limit=limit)
    out_dir = os.path.join(SRC_PATH, f'out-{PATIENT}{tag}')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    os.chdir(out_dir)

    # Create summary.txt
    with open(os.path.join(os.getcwd(), 'summary.txt'), 'w') as f:  # Opens file and casts as f
        f.write('Summary statistics for concatenated EDF files:\n\n')

    # Export one merged file for each continuous time-interval.
    for i, continuous_interval in enumerate(edf_files):
        if len(continuous_interval) <= 1:
            continue
        # Verbose for testing
        write_txt(f'Interval {i} Data:')
        merged = continuous_interval.files.pop(0)
        merged = trim_and_decimate(to_edf(merged), 200)
        for edf in continuous_interval.files:
            res = trim_and_decimate(to_edf(edf), 200)
            write_txt(str(res.info))
            write_txt(f'{res.get_data().shape[0]} x {res.get_data().shape[1]}')
            merged = concatenate(merged, res)

        # One liner -> trimmed: list[mne.io.Raw] = [trim_and_decimate(to_edf(edf), 200) for edf in continuous_interval]
        #merged: mne.io.Raw = concatenate(trimmed)
        out_name: str = f'{name}-{i}{tag}'
        export(merged, out_name, 'bipolar')
        write_txt(f'{out_name} Data:\n', str(merged.info), f'Total concatenated: {len(continuous_interval)}',
                  f'{merged.get_data().shape[0]} x {merged.get_data().shape[1]}', str(merged.get_data()))
        assert f'{out_name}.edf' in os.listdir(), f'Export failed. {out_name}.edf not in {os.listdir()}.'
