'''
Constructs RV data from pickle file resulting from `download.py`.

File has the following structure:
{
    `apstar_id_1` : {
        'jd' : np.array [1, 2, ...],
        'vhelio' : np.array [1, 2, ...]
        },
    ...
}
'''

import matplotlib.pyplot as plt

import pickle
import numpy as np
import argparse

def load_data(filename: str) -> dict:
    '''
    Loads data from pickle file.
    '''
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def remove_999(data: np.ndarray) -> np.ndarray:
    '''
    Removes 999 from data
    '''
    new_data = data.copy().astype(float)
    trigger = new_data == -9999.
    new_data[trigger] = np.nan
    return new_data

def _parseargs():
    '''
    Parses arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=int)
    parser.add_argument("--filename", type=str, default="data.pkl")
    return parser.parse_args()

if __name__ == "__main__":
    '''
    Visualizes RV data
    '''
    args = _parseargs()

    ### Parsing data code
    data = load_data(args.filename)
    print(f"Total number of apstar_ids: {len(data.keys())}")

    ### Plotting code
    slicer = args.slice # Push forward window by this amoutn
    idx_pluser = (4 * 4) * slicer

    fiducial_jd = 2457388.5

    # Plot data in a 4 x 4 grid
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))

    for idx, ax in enumerate(axes.flat):
        # Get `apstar_id`
        apstar_id = list(data.keys())[idx + idx_pluser]

        # Get data
        entry = data[apstar_id]
        jd = entry['jd']
        vhelio = remove_999(entry['vhelio'])
        vrelerr = remove_999(entry['vrelerr'])

        ax.errorbar(jd - fiducial_jd, vhelio, yerr=vrelerr, color='black', linestyle=' ', marker='o')
        ax.set_title(f'{apstar_id}', fontsize=8)
        ax.set_xlabel(f'JD - {fiducial_jd}')
        ax.set_ylabel('vhelio')

    plt.tight_layout()
    plt.show()
    
    
