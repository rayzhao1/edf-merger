# edf-merger

Python script to merge a directory of EDF files into a single EDF file.

## Dependencies:
  1) mne 1.5.1    
  2) Python 3.10+    
  3) EDFlib-Python-1.0.8+
    
## Usage:
  python3 edf_merge.py <edf-file-path> <[optional] output-file-name> <[Merge-checking (optional)] False>
    - The output file is created in the current working directory, in a folder titled 'out'.

## Requirements:
  1) The <edf-file-path> directory stores the EDF files to be merged in a folder with the same name.
  2) If "Merge-checking" is on, the <edf-file-path> directory contains a '<name>_EDFMeta.csv'.

## Example - Valid File Structure:
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
