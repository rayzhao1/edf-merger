"""
Usage (all absolute paths):
    python edf_merge_pr03.py <edf-files-dir-path> <edf-catalog-path> <output-dir-path>

Example - Default File Structure:
    .
    └── data_store0
        └── presidio
            └── nihon_kohden
                └── PR03
                    ├── PR03_EDFMeta.csv
                    └── PR03
                        ├── PR03_2605.edf
                        ├── PR03_2606.edf
                        └── PR03_2607.edf
                        
    * able to avoid absolute paths only because expected file structure is used            
        python edf_merge_pr03.py ~/ray/data_store0/presidio/nihon_kohden/PR03 ~/ray/data_store0/presidio/nihon_kohden/PR03/PR03_EDFMeta.csv out-PR03 --local
"""
from pathlib import Path
import argparse
import shutil
import mne
from mne.io import read_raw_edf
from mne.export import export_raw
import csv
import datetime
from datetime import timedelta
import time
from scipy.signal import detrend

NIGHT_START_HOUR: int = 21  # 9pm
NIGHT_DURATION: timedelta = timedelta(hours=11)  # hours=11)


class Night:
    def __init__(self):
        self.intervals: list[Interval] = []

    def add(self, file):
        self.intervals.append(file)


class Interval:
    def __init__(self, t0=None, tf=None):
        self.files: list[str] = []
        self.t0 = t0
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def add(self, file):
        self.files.append(file)


def scalp_trim_and_decimate(raw_edf: mne.io.Raw, freq: int) -> mne.io.Raw:
    """Takes an EDF file path and returns the EDF data as an mne.io.Raw object with:
        - Only scalp channels included.
        - Resamples input EDF's frequency to 'freq'.
    """
    rename_dict: dict[str: str] = {name: name[4:] for name in raw_edf.ch_names}
    if "POL L EMG-Ref" in rename_dict:
        rename_dict["POL L EMG-Ref"] = 'L_EMG-Ref'
    if "POL R EMG-Ref" in rename_dict:
        rename_dict["POL R EMG-Ref"] = 'R_EMG-Ref'
    if 'POL L EOG-Ref' in rename_dict:
        rename_dict['POL L EOG-Ref'] = 'L_EOG-Ref'
    if 'POL R EOG-Ref' in rename_dict:
        rename_dict['POL R EOG-Ref'] = 'R_EOG-Ref'

    raw_edf = raw_edf.rename_channels(rename_dict)
    # Remove non scalp-eeg
    channels: list[str] = raw_edf.ch_names
    scalp_start: int = channels.index('Fp1-Ref')
    to_drop = channels[:scalp_start]
    raw_edf_scalp = raw_edf.drop_channels(to_drop)
    print(raw_edf_scalp.ch_names)
    # Decimate 2000 hz to 200 hz
    raw_edf_scalp = raw_edf_scalp.resample(freq)
    return raw_edf_scalp


def scalp_bipolar_reference(raw_edf: mne.io.Raw) -> mne.io.Raw:
    cathodes = ['Fp1-Ref', 'F7-Ref', 'T7-Ref', 'P7-Ref', 'Fp1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'Fz-Ref', 'Cz-Ref',
                'Fp2-Ref', 'F4-Ref', 'C4-Ref', 'P4-Ref', 'Fp2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref', 'L_EOG-Ref',  'R_EOG-Ref', 'L_EMG-Ref']
    anodes   = ['F7-Ref', 'T7-Ref', 'P7-Ref', 'O1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'O1-Ref', 'Cz-Ref', 'Pz-Ref',
                'F4-Ref', 'C4-Ref', 'P4-Ref', 'O2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref', 'O2-Ref', 'A2-Ref', 'A1-Ref', 'R_EMG-Ref']
    names    = ['Fp1_F7', 'F7_T7', 'T7_P7', 'P7_O1', 'Fp1_F3', 'F3_C3', 'C3_P3', 'P3_O1', 'Fz_Cz', 'Cz_Pz',
             'Fp2_F4', 'F4_C4', 'C4_P4', 'P4_O2', 'Fp2_F8', 'F8_T8', 'T8_P8', 'P8_O2', 'L-EOG_A2', 'R-EOG_A1', 'L-EMG_R-EMG']
    assert len(cathodes) == len(anodes) == len(names), 'Incorrect cathodes, anodes, names input to bipolar_reference()'
    return mne.set_bipolar_reference(raw_edf, anodes, cathodes, names)


def export_edf(raw_edf: mne.io.Raw, target_name: str, mode=None, overwrite_existing=True):
    """Export raw object as EDF file"""
    name: str = f'{target_name}.edf'
    match mode:
        case 'bipolar':
            res = scalp_bipolar_reference(raw_edf)
        case 'common_average':
            res = raw_edf.set_eeg_reference()
        case 'bipolar_common_average':
            res = scalp_bipolar_reference(raw_edf).set_eeg_reference()
        case _:
            res = raw_edf
    export_raw(name, res, 'edf', overwrite=overwrite_existing)
    return res


def str_to_time(time_str: str, time_format='%Y-%m-%d %H:%M:%S') -> datetime.datetime:
    if '.' in time_str:
        return datetime.datetime.strptime(time_str.split('.')[0], time_format)
    elif '+' in time_str: # DP01
        return datetime.datetime.strptime(time_str.split('+')[0], time_format)
    else:
        raise Exception(f'Invalid time_str {time_str}')


def get_first_date(csv_in: str, date_idx=3) -> datetime.datetime:
    with open(csv_in) as csv_file:
        csv_reader: csv.reader = csv.reader(csv_file, delimiter=',')
        _, first_row = next(csv_reader), next(csv_reader)
        return str_to_time(first_row[date_idx])


def round_day(dt):
    if dt.hour >= 12:
        dt = (dt + timedelta(days=.5)).replace(hour=0, minute=0, second=0, microsecond=0)
    return dt


def parse_nights(
    csv_in: str, 
    key: dict[str, str], 
    all_files=None, 
    idx=None, 
    margin=timedelta(seconds=0),
    strict=True,
) -> list[Night]:
    """Iterate through 'csv_in' and return a list of lists, where each sublist contains an contiguous_interval of EDF file names
       such that each EDF is less than 'margin' away from the previous file in time. This implementation relies on the
       fact that csv_in is sorted in time-chronological order. All returned EDF files are also constrained to be in the
       time range between 'start' and 'end'.
    """
    start_idx, end_idx = key['start'], key['end']
    edf_name_idx, edf_path_idx = key['edf_name'], key['edf_path']
    ch_name_idx = key.get('ch_names')

    start: datetime.datetime = get_first_date(csv_in, date_idx=start_idx)  # Set start date
    start = start.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)  # Set start date and time
    end: datetime.datetime = start + NIGHT_DURATION  # Set initial end time

    nights: list[Night] = []
    curr_night, curr_interval = Night(), Interval()

    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # remove header

        for row in csv_reader:
            curr_name, curr_path = row[edf_name_idx], row[edf_path_idx]
            curr_time_start, curr_time_end = str_to_time(row[start_idx]), str_to_time(row[end_idx])
            # (1) Do not record files earlier than start, or file names that cannot be found in folder.
            if curr_time_start < start:
                prev_time_end = curr_time_end
                continue
            # (2) If we've reached the end of the night, set 'tf', add this night and prepare for the next
            if curr_time_start >= end:
                curr_interval.tf = prev_time_end
                curr_night.add(curr_interval)
                nights.append(curr_night)
                curr_interval = Interval()
                curr_night = Night()
                # Update [start, end] time interval for next night
                start = round_day(curr_time_start).replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)  # Set start date and time
                end = start + NIGHT_DURATION
                continue
            # (3) We've reached a new interval, only set 't0' for the first file in an interval!
            if not curr_interval.files:
                curr_interval.t0 = curr_time_start
            # (4) Sanity check - does EDF meta file actually exist in directory
            if all_files and curr_name not in all_files:
                if not strict:
                    print(f'Warning - {curr_name} not found in EDF directory.')
                    continue
                else:
                    raise Exception(f'File {curr_name} not found in directory.')
            # (5) Sanity check - Very rare (PR03), some files do not not have scalp data
            if ch_name_idx and 'POL Fp1-Ref' not in eval(row[ch_name_idx]):
                continue
            # (6) If there is a discontinuity, create a new Interval for the current night.
            if curr_time_start - prev_time_end > margin:
                if curr_interval.files: # make sure the discontinuity exists with a file that is in the interval
                    print(f'Discontinuity at {curr_name}')
                    curr_interval.tf = prev_time_end
                    curr_night.add(curr_interval)
                    curr_interval = Interval(t0=curr_time_start)
            # (7) Finally, we're safe to add the data at 'idx' to current interval, or the file path by default
            if idx:
                vals = tuple([curr_path] + [row[i] for i in idx])
                curr_interval.add(vals)
            else:
                curr_interval.add(row[edf_path_idx])
            prev_time_end = curr_time_end
        # Tail case
        if curr_interval.files:
            curr_interval.tf = prev_time_end
            curr_night.add(curr_interval)
            nights.append(curr_night)
    
    return nights


def verify_night_ranges(patient, nights):
    """Hard-coded verification. Ranges are (inclusive, exclusive]"""
    def get_num_str(num):
        if num < 100:
            return f'{patient}_00{num}.edf'
        if num < 1000:
            return f'{patient}_0{num}.edf'
        else:
            return f'{patient}_{num}.edf'
    
    def verify_night(night_num, expected_range, expected_len=None):
        correct_len = len(nights[night_num].intervals[0].files) == expected_len if expected_len else True
        correct_files = [Path(fn).name for fn in nights[night_num].intervals[0].files] == [get_num_str(i) for i in expected_range]
        if not correct_len:
            raise Exception(f'Expected {expected_len}, got length {len(nights[night_num].intervals[0].files)}')
        if not correct_files:
            raise Exception(f'Expected {[get_num_str(i) for i in expected_range]}, got files {[Path(fn).name for fn in nights[night_num].intervals[0].files]}')

    def verify_dp01():
        """Total 9 nights."""
        exp = (
            range(106, 238),
            range(387, 519),
            range(667, 799),
            range(951, 1083),
            range(1237, 1369),
            range(1521, 1653),
            range(1806, 1938),
            range(2089, 2221),
        )
        for i, exp in enumerate(exp):
            verify_night(i, exp, expected_len=132)

    def verify_pr03():
        """total 10 nights, missing data for night 6"""
        exp = (
            range(40, 172),
            range(326, 458),
            range(612, 744),
            range(898, 1030),
            range(1171, 1303),
            None,               # Interval has no scalp data
            None,               # range(1629, 1761),
            None,               # range(1914, 2046),
            None,               # range(2200, 2332),
            None,               # range(2481, 2613),
        )
        for i, exp in enumerate(exp):
            if exp:
                verify_night(i, exp)
        
    def verify_pr05():
        """Total 9 nights, no discontinuities or breaks."""
        exp = (
            range(129, 261),
            range(415, 547),
            range(697, 829),
            range(982, 1114),
            range(1260, 1392),
            range(1545, 1677),
            range(1830, 1962),
            range(2114, 2246),
            range(2399, 2531),
        )
        for i, exp in enumerate(exp):
            verify_night(i, exp, expected_len=132)

    match patient:
        case 'DP01':
            verify_dp01()
        case 'PR03':
            verify_pr03()
        case 'PR05':
            verify_pr05()
        case _:
            raise Exception(f'Unexpected patient ID: {patient}')
    print("Ranges were successfully verified!")


def get_patient_csv_key(patient_id: str):
    match patient_id:
        case 'DP01':
            key = dict(
                start=1,
                end=2,
                edf_name=0,
                edf_path=5,
                ch_names=None,
            )
        case 'PR03':
            # 0:_ 1:edf_path_short	2:edf_path	3:edf_start	4:edf_end	5:edf_length	6:edf_sfreq	7:edf_chnames
            key = dict(
                start=3,
                end=4,
                edf_name=1,
                edf_path=2,
                ch_names=7,
            )
        case 'PR05':
            raise NotImplementedError
        case 'PR06':
            raise NotImplementedError
        case _:
            raise Exception(f'Unexpected patient ID: {patient_id}')
    return key


def resolve_args(args, strict=True):
    def is_path(s): # not robust
        return '/' in s or '\\' in s
    def rmdir(dir):
        shutil.rmtree(dir)

    script_dir = Path(__file__).resolve(strict=True).parent
    
    if not is_path(args.edfs_dir):
        patient_name = args.edfs_dir
        patient_dir = Path(f'/data_store0/presidio/nihon_kohden/{patient_name}') # fix 
        catalog_path = patient_dir.joinpath(f'{Path(args.catalog_path).stem}.csv')
    else:
        patient_dir = Path(args.edfs_dir)
        patient_name = patient_dir.name
        catalog_path = args.catalog_path

    edfs_dir = patient_dir.joinpath(patient_name)
    
    output_dir = script_dir.joinpath(args.output_dir) if not is_path(args.output_dir) else Path(args.output_dir)
    rmdir(output_dir) if output_dir.exists() else None
    output_dir.mkdir(exist_ok=False)

    if strict:
        assert patient_dir.exists(), f'Specified EDF directory does not exist: {patient_dir}.'
        assert edfs_dir.exists(), f'Specified EDF directory does not have a {patient_name} folder.'
        assert output_dir.exists(), f'Specified EDF directory does not have a {patient_name} folder.'
        assert catalog_path.exists(), f'Specified EDF metadata catalog does not exist at {catalog_path}.'
    
    return patient_name, dict(
        script=script_dir,
        patient=patient_dir,
        edfs=edfs_dir,
        output=output_dir,
        meta=catalog_path,
    )

def get_server_run_inputs(patient):
    match patient:
        case 'PR03':
            return ['PR03', 'PR03_EDFMeta', 'out-PR03-3.28']
        case 'DP01':
        # python edf_merge_dp01.py ~/ray/data_store2/OCD_SEEG/nihon_kohden/DP01 ~/ray/data_store2/OCD_SEEG/nihon_kohden/DP01/nkhdf5/DP01_edf_catalog.csv out-DP01-4.1 -l
            return [
                '/data_store2/OCD_SEEG/nihon_kohden/DP01', 
                '/data_store2/OCD_SEEG/nihon_kohden/DP01/nkhdf5/DP01_edf_catalog',
                'out-DP01-4.1',
                ]


if __name__ == "__main__":
    timer_start = time.time()

    PATIENT = 'DP01' # Must set if on server

    # Process command-line args
    parser = argparse.ArgumentParser(description='EDF Merger')

    parser.add_argument("edfs_dir",
                        help='absolute path to directory containing EDFs to merge, or a patient name')
    parser.add_argument("catalog_path",
                        help='absolute path to directory containing EDF metadata catalog, or catalog name')
    parser.add_argument('output_dir',
                        help='absolute path to directory to store merged output in, or a string name')
    parser.add_argument("-l", "--local", action="store_true",
                        help='flag to indicate that script is run locally; default is run on server')
    try: # (1) If local use, actually parse arguments
        args = parser.parse_args()
    except SystemExit: # (2) If server use, script must be run as job so hardcode arguments
        args = parser.parse_args(get_server_run_inputs(PATIENT))

    patient_name, path = resolve_args(args, strict=not args.local)

    # Retrieve list of sub lists. Each sublist is a set of continuous, in-range file names.
    all_edfs = set([p.name for p in path['edfs'].iterdir()]) if not args.local else None
    key = get_patient_csv_key(patient_name)

    nights: list[Night] = parse_nights(
        path['meta'], 
        key, 
        all_files=all_edfs, 
        strict=not args.local,
    )
    if not args.local: # Only perform verification if on server
        verify_night_ranges(patient_name, nights)
    
    # For each night, export one merged file for each continuous time-interval for each night.
    for night_num, night in enumerate(nights):
        for interval_num, interval in enumerate(night.intervals):
            # If `interval.t0 is None`, then that interval never reached a starting point.
            if not interval.files or not interval.t0:
                continue
            t0_str = interval.t0.strftime("%Y-%m-%d_%H.%M")
            tf_str = interval.tf.strftime("%Y-%m-%d_%H.%M")
            out_path = path['output'].joinpath(f'{patient_name}_night_{night_num+1}.{interval_num+1}_scalp_{t0_str}--{tf_str}')
            interval_files = interval.files if not args.local else [path['edfs'].joinpath(Path(f).name) for f in interval.files]
            if True:
                print(out_path)
                print(len(interval_files))
                continue
            concatenated = mne.concatenate_raws([scalp_trim_and_decimate(read_raw_edf(edf_path), 200) for edf_path in interval_files])
            # 1) bandpass for neural data 2) bandstop for electrical noise 3) demean 4) scale          
            res = (concatenated
                    .filter(l_freq=0.5, h_freq=80, verbose=False)
                    .notch_filter(60, notch_widths=4)
                    .apply_function(detrend, channel_wise=True, type="constant")
                    .apply_function(lambda x: x*1e-6, picks="eeg")
            )
            export_edf(res, out_path, mode='bipolar', overwrite_existing=True)

    timer_end = time.time()
    print(f'Time elapsed: {timer_end - timer_start}')
