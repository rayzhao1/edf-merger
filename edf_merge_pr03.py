"""
Usage (all absolute paths):
    python edf_merge_pr03.py ~/ray/ucsf/data_store0/presidio/nihon_kohden/PR03 ~/ray/ucsf/data_store0/presidio/nihon_kohden/PR03/PR03_EDFMeta.csv out-PR03-5.1 --local
    submit_job -q pia-batch.q -c 8 -m 40 -o /userdata/rzhao/results.txt -n pr03 -x /userdata/rzhao/newenv/bin/python3 /userdata/rzhao/edf-merger/edf_merge_pr03.py

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
"""
from pathlib import Path
import argparse
import shutil

import csv
import datetime
from datetime import datetime, timedelta, timezone
import time
from itertools import chain
from functools import reduce

import numpy as np
from scipy.signal import detrend
import mne
from mne.io import read_raw_edf
from mne.export import export_raw
from edfio import read_edf


NIGHT_START_HOUR: int = 21  # 9pm
NIGHT_DURATION: timedelta = timedelta(hours=11)  # hours=11)
S_FREQ = 200
CHANNELS: dict[str, tuple[str]] = dict(
    eeg=('Pz-Ref', 'A1-Ref', 'O1-Ref', 'F3-Ref', 'T8-Ref', 'F8-Ref', 'A2-Ref',
        'F4-Ref', 'P7-Ref', 'P3-Ref', 'T7-Ref', 'F7-Ref', 'Fp1-Ref', 'P8-Ref',
        'Fz-Ref', 'O2-Ref', 'C4-Ref', 'P4-Ref', 'Fp2-Ref', 'C3-Ref', 'Cz-Ref',
        ),
    eog=('L_EOG-Ref', 'R_EOG-Ref'),
    emg=('L_EMG-Ref', 'R_EMG-Ref'),
    ecg=('L_EKG-Ref', 'R_EKG-Ref'),
)

class Night:
    night_count = 0
    def __init__(self):
        self.idx = Night.night_count
        self.intervals: list[Interval] = []
        self.interval_count = 0
        Night.night_count += 1

    def add(self, interval):
        self.intervals.append(interval)
        interval.idx = self.interval_count
        self.interval_count += 1


class Interval:
    def __init__(self, t0=None, tf=None):
        self.files: list[str] = []
        self.t0 = t0
        self.tf = tf

    def add(self, file):
        self.files.append(file)
    
    def edf_pad_and_crop(
            self, 
            interval_files, 
            channel_dict,
    ):
        curr_edf_path = interval_files.pop(0)
        curr_edf = read_raw_edf_corrected(curr_edf_path, channel_dict)
        processed_edfs = []
        for next_edf_path in interval_files:
            print(f"Processing {curr_edf_path} and {next_edf_path}")
            next_edf = read_raw_edf_corrected(next_edf_path, channel_dict)
            curr_end, next_start = curr_edf.info['meas_date']+timedelta(seconds=curr_edf.duration), next_edf.info['meas_date']

            if curr_end < next_start:
                curr_edf = edf_pad(curr_edf, next_start-curr_end)
            elif curr_end > next_start:
                curr_edf = edf_crop(curr_edf, curr_end-next_start)

            processed_edfs.append(curr_edf)
            curr_edf = next_edf
        processed_edfs.append(next_edf) # tail case
        return processed_edfs
    
    def edf_concatenate(
            self,  
            channel_dict: dict[str, str],
            path: dict[str, str],
            expected_seconds: int=NIGHT_DURATION.seconds,
            local: bool=False,
            s_freq: int=S_FREQ,
    ):
        """Concatenates self.files as a continuous EDF file. Each individual file is processed according to `preproc`,
           and the resulting concatenated file is processed according to `postproc`.
        """
        interval_files = list(self.files) if not local else [path['edfs'].joinpath(Path(f).name) for f in self.files]
        first = read_raw_edf(interval_files[0])
        assert abs(first.info['meas_date'] - self.t0) < timedelta(seconds=1), f"\nfirst.info['meas_date']=={first.info['meas_date']}\nself.t0={self.t0}"

        aligned_edfs = self.edf_pad_and_crop(interval_files, channel_dict)
        preprocessed = [edf.resample(s_freq) for edf in aligned_edfs]
        res = postprocess_night_edf(mne.concatenate_raws(preprocessed), channel_dict)

        assert abs(res.info['meas_date'] - self.t0) < timedelta(seconds=1), f"\nres.info['meas_date']={res.info['meas_date']}\nself.t0={self.t0}"
        # assert abs(res.duration - expected_seconds) < 5, f"\nres.duration={res.duration}\nexpected_seconds={expected_seconds}"
        assert res.n_times == int(res.duration * S_FREQ)

        return res


def postprocess_night_edf(edf, channel_dict):
    # 1) bandpass for neural data 2) bandstop for electrical noise 3) demean 4) scale 
    chtypes = set([channel_dict[k][1] for k in channel_dict])
    return (edf
                .filter(l_freq=0.5, h_freq=80, verbose=False)
                .notch_filter(60, notch_widths=4)
                .apply_function(detrend, channel_wise=True, type="constant")
                .rescale({k:v for k,v in dict(
                    eeg=1e-6,
                    eog=1e-6,
                    emg=1e-7,
                    ecg=1e-7,
                ).items() if k in chtypes})
        )


def edf_pad(raw_edf, amount):
    curr_data = raw_edf.get_data()  # Last time point in seconds
    sfreq = raw_edf.info['sfreq']
    old = raw_edf.duration
    pad_samples = int(amount.seconds * sfreq)

    pad_data = np.zeros((len(raw_edf.ch_names), pad_samples))
    new_data = np.hstack((curr_data, pad_data))

    # Create new info object
    new_info = raw_edf.info.copy()
    res = mne.io.RawArray(new_data, new_info)
    print(f"Attempted to pad {amount}: From {old} to {res.duration}.")
    return res


def edf_crop(raw_edf, amount):
    amount_sec = amount.microseconds * 1e-6 # Raw.crop wants seconds, and .seconds kills subsecond resolution
    old = raw_edf.duration
    res = raw_edf.copy().crop(tmin=0, tmax=raw_edf.duration-amount_sec, include_tmax=False)
    print(f"Attempted to crop {amount_sec} seconds: From {old} to {res.duration}.")
    return res


def read_raw_edf_corrected(fn, channel_dict) -> mne.io.Raw:
    """For whatever reason, mne.io.read_raw_edf zeros an input file's subseconds. Pyedflib does not
       better, so the edfio library is needed.
    """
    chnames, ch_to_rename, ch_to_type = [], {}, {}
    for ch, (chname, chtype) in channel_dict.items():
        chnames.append(ch)
        ch_to_rename[ch], ch_to_type[ch] = chname, chtype
    # (1) Adjust start date
    edf_edfio = read_edf(fn)
    start = datetime.combine(edf_edfio.startdate, edf_edfio.starttime).replace(tzinfo=timezone.utc)
    edf_mne = read_raw_edf(fn, include=chnames)
    old = edf_mne.info['meas_date']
    edf_mne.set_meas_date(start)
    print(f"Adjusted EDF {fn} start from {old} to {edf_mne.info['meas_date']}")
    # (2) Set channel type for each channel, Standardize channel names
    return edf_mne.set_channel_types(ch_to_type).rename_channels(ch_to_rename)


def scalp_bipolar_reference(raw_edf: mne.io.Raw, channel_dict) -> mne.io.Raw:
    cathodes = ['Fp1-Ref', 'F7-Ref', 'T7-Ref', 'P7-Ref', 'Fp1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'Fz-Ref', 'Cz-Ref',
                'Fp2-Ref', 'F4-Ref', 'C4-Ref', 'P4-Ref', 'Fp2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref', 'L_EOG-Ref',  'R_EOG-Ref', 'L_EMG-Ref', 'L_EKG-Ref']
    anodes   = ['F7-Ref', 'T7-Ref', 'P7-Ref', 'O1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'O1-Ref', 'Cz-Ref', 'Pz-Ref',
                'F4-Ref', 'C4-Ref', 'P4-Ref', 'O2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref', 'O2-Ref', 'A2-Ref', 'A1-Ref', 'R_EMG-Ref', 'R_EKG-Ref']
    names    = ['Fp1_F7', 'F7_T7', 'T7_P7', 'P7_O1', 'Fp1_F3', 'F3_C3', 'C3_P3', 'P3_O1', 'Fz_Cz', 'Cz_Pz',
                'Fp2_F4', 'F4_C4', 'C4_P4', 'P4_O2', 'Fp2_F8', 'F8_T8', 'T8_P8', 'P8_O2', 'L-EOG_A2', 'R-EOG_A1', 'L-EMG_R-EMG', 'L-EKG_R-EKG']
    to_drop = set()
    existing_chs = set([channel_dict[k][0] for k in channel_dict])
    for grp in [cathodes, anodes]:
        for i, ch in enumerate(grp):
            if ch not in existing_chs:
                to_drop.add(i)
    for i in reversed(sorted(to_drop)):
        cathodes.pop(i), anodes.pop(i), names.pop(i)

    assert len(cathodes) == len(anodes) == len(names), 'Incorrect cathodes, anodes, names input to bipolar_reference()'
    return mne.set_bipolar_reference(raw_edf, anodes, cathodes, names, drop_refs=True).pick(names) #, names)


def export_edf(raw_edf: mne.io.Raw, channel_info, target_name: str, mode=None, overwrite_existing=True):
    """Export raw object as EDF file. Result is zero-padded if non-integer duration (MNE behavior)"""
    fname: str = f'{target_name}.edf'
    match mode:
        case 'bipolar':
            res = scalp_bipolar_reference(raw_edf, channel_info)
        case 'common_average':
            res = raw_edf.set_eeg_reference()
        case 'bipolar_common_average':
            res = scalp_bipolar_reference(raw_edf, channel_info).set_eeg_reference()
        case _:
            res = raw_edf
    export_raw(fname, res, fmt='edf', overwrite=overwrite_existing)
    return res


def str_to_time(time_str: str, time_format='%Y-%m-%d %H:%M:%S') -> datetime:
    if '.' in time_str:
        return datetime.strptime(time_str.split('.')[0], time_format).replace(tzinfo=timezone.utc)
    elif '+' in time_str: # DP01
        return datetime.strptime(time_str.split('+')[0], time_format).replace(tzinfo=timezone.utc)
    else:
        raise Exception(f'Invalid time_str {time_str}')


def get_first_date(csv_in: str, date_idx=3) -> datetime:
    with open(csv_in) as csv_file:
        csv_reader: csv.reader = csv.reader(csv_file, delimiter=',')
        _, first_row = next(csv_reader), next(csv_reader)
        print(first_row)
        return str_to_time(first_row[date_idx])


def round_day(dt):
    if dt.hour >= 12:
        dt = (dt + timedelta(days=.5)).replace(hour=0, minute=0, second=0, microsecond=0)
    return dt


def csv_parse_nights(
    csv_in: str, 
    key: dict[str, str], 
    all_files,
    night_start_hour: int=NIGHT_START_HOUR,
    night_duration: timedelta=NIGHT_DURATION, 
    idx=None, 
    margin=timedelta(seconds=0),
    local=True,
) -> list[Night]:
    """Iterate through 'csv_in' and return a list of Nights, where each Night contains a list of contiguous interval of EDF file
       names such that each EDF is less than 'margin' away from the next file in time. This implementation relies on the
       fact that csv_in is sorted in time-chronological order. All returned EDF files are also constrained to be in the
       time range between 'start' and 'start + duration'.
    """
    strict = not local
    start_idx, end_idx = key['start'], key['end']
    edf_name_idx, edf_path_idx = key['edf_name'], key['edf_path']
    ch_name_idx = key.get('ch_names')

    start: datetime = get_first_date(csv_in, date_idx=start_idx)  # Set start date
    start = start.replace(hour=night_start_hour, minute=0, second=0, microsecond=0)  # Set start date and time
    end: datetime = start + night_duration  # Set initial end time

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
                start = round_day(curr_time_start).replace(hour=night_start_hour, minute=0, second=0, microsecond=0)  # Set start date and time
                end = start + night_duration
                continue
            # (3) We've reached a new interval, only set 't0' for the first file in an interval!
            if not curr_interval.files:
                curr_interval.t0 = curr_time_start
            # (4) Sanity check - does EDF listed in catalog actually exist in directory
            if curr_name not in all_files:
                if strict:
                    raise Exception(f'File {curr_name} not found in directory.')
                else:
                    print(f'Warning - {curr_name} not found in EDF directory.')
                    continue
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
            else:
                vals = row[edf_path_idx]
            curr_interval.add(vals)
            prev_time_end = curr_time_end
        # Tail case
        if curr_interval.files:
            curr_interval.tf = prev_time_end
            curr_night.add(curr_interval)
            nights.append(curr_night)

    if not local: # Only perform verification if on server
        verify_night_ranges(patient_name, nights)

    return nights


def verify_night_ranges(patient, nights):
    """Hard-coded verification. Ranges are (inclusive, exclusive]"""
    
    def verify_night(night_num, expected_range, expected_len=None):
        correct_len = len(nights[night_num].intervals[0].files) == expected_len if expected_len else True
        correct_files = [Path(fn).name for fn in nights[night_num].intervals[0].files] == [f'{patient}_{i:04}.edf' for i in expected_range]
        if not correct_len:
            raise Exception(f"Expected {expected_len}, got length {len(nights[night_num].intervals[0].files)}")
        if not correct_files:
            raise Exception(f"Expected {[f'{patient}_{i:04}.edf' for i in expected_range]}, got files {[Path(fn).name for fn in nights[night_num].intervals[0].files]}")

    def verify_dp01():
        """Total 8 nights."""
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
        """total 10 nights, missing data for night 5"""
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
        catalog_path = Path(args.catalog_path)

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
            return ['PR03', 'PR03_EDFMeta', 'out-PR03-5.1']
        case 'DP01':
            return [
                '/data_store2/OCD_SEEG/nihon_kohden/DP01', 
                '/data_store2/OCD_SEEG/nihon_kohden/DP01/nkhdf5/DP01_edf_catalog.csv',
                '/userdata/rzhao/out-DP01-4.30',
                ]

def get_channel_info(patient, channels: dict[str, str]):
    """Return a mapping
        Name of channel to be selected
            to
        tuple[ Final channel name to rename to , Channel type ]
    """
    lemg, remg = 'L_EMG-Ref', 'R_EMG-Ref'
    leog, reog = 'L_EOG-Ref', 'R_EOG-Ref'
    lekg, rekg = 'L_EKG-Ref', 'R_EKG-Ref'
    match patient:
        case 'DP01':
            # newname_to_oldname, None to indicate channel doesn't exist.
            new_to_old={
                lemg: 'LEMG1-Ref', remg: 'REMG1-Ref',
                leog: 'LEOG-Ref', reog: 'REOG-Ref',
                lekg: 'L_EKG-Ref', rekg: 'R_EKG-Ref'
            }
        case 'PR03':
            new_to_old={
                lemg: 'L EMG-Ref', remg: 'R EMG-Ref',
                leog: 'L EOG-Ref', reog: 'R EOG-Ref',
                lekg: None, rekg:  None
            }
        case 'PR07':
            new_to_old={
                lemg: 'EMG1-Ref', remg: 'EMG2-Ref',
                leog: 'L EOG-Ref', reog: 'R EOG-Ref',
                lekg: 'EKG1-Ref', rekg:  'EKG2-Ref'
            }
        case _:
            raise Exception(f'Unexpected patient ID: {patient}')
        
    scalp_ch = dict()
    for chtype, chnames in channels.items():
        for chname in chnames:
            found = False
            for new, old in new_to_old.copy().items():
                if chname == new and old != None:
                    # del scalp_ch[f'POL {old}']
                    del new_to_old[new]
                    scalp_ch[f'POL {old}'] = (new, chtype)
                    found = True
                    break
            if not found:
                if chname in new_to_old and not new_to_old.get(chname):
                    del new_to_old[chname]
                    continue
                scalp_ch[f'POL {chname}'] = (chname, chtype)
            else:
                found = False
    return scalp_ch


if __name__ == "__main__":
    timer_start = time.time()

    PATIENT = 'PR03' # Must set if on server
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
    # (2) Retrieve list of sub lists. Each sublist is a set of continuous, in-range file names.
    nights: list[Night] = csv_parse_nights(
        csv_in=path['meta'], 
        key=get_patient_csv_key(patient_name), 
        all_files=set([p.name for p in path['edfs'].iterdir()]), 
        local=args.local,
    )
    # (3) For each night, export one merged file for each continuous time-interval for each night.
    for night in nights:
        for interval in night.intervals:
            # If `interval.t0 is None`, then that interval never reached a starting point.
            if not interval.files or not interval.t0:
                continue
            t0_str, tf_str = interval.t0.strftime("%Y-%m-%d_%H.%M"), interval.tf.strftime("%Y-%m-%d_%H.%M")
            out_path = path['output'].joinpath(f'{patient_name}_night_{night.idx+1}.{interval.idx+1}_scalp_{t0_str}--{tf_str}')
            channel_info = get_channel_info(patient_name, CHANNELS)
            res = interval.edf_concatenate(
                channel_dict=channel_info, 
                path=path, 
                local=args.local
            )
            export_edf(res, channel_info, out_path, mode='bipolar', overwrite_existing=True)

    timer_end = time.time()
    print(f'Time elapsed: {timer_end - timer_start}')
