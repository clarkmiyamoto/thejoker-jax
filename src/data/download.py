from astroquery.sdss import SDSS

import numpy as np

import argparse
import pickle
from tqdm import tqdm

def get_apstarId(visits: int=7, top: int=50):
  '''
  Args:
    - count: Star must have more than `visits` number of visits

  Returns:
    - res. Has columns 'apogee_id', 'apstar_id'
  '''
  query = f"""
  SELECT TOP {str(top)} s.apogee_id, s.apstar_id, s.nvisits
  FROM apogeeStar AS s
  WHERE s.nvisits > 7
  """
  res = SDSS.query_sql(query, data_release=19)

  return res

def get_visitIds_from_apstarId(apstar_id: str):
  '''
  Gets visit_id from apstar_id

  Args:
    - apstar_id: str

  Returns:
    - res. Has columns 'visit_id', 'apstar_id'

  Assumptions:
    - SDSS Data Release 19
  '''
  query = f"""
  SELECT visit_id, apstar_id
  FROM apogeeStarAllVisit
  WHERE apstar_id = "{apstar_id}"
  """
  res = SDSS.query_sql(query, data_release=19)
  return res

def get_rvs_from_visit_id(visit_id: str):
  """
  Gets radial velocity (RV_data) from visit_id

  Documentation on query: https://skyserver.sdss.org/dr18/MoreTools/browser

  We are using the `apogeeVisit` table.
  It has the following properties:
  - `fiberid` (int)
  - `plate` (int)
  - `mjd` (float): days
  - `jd` (float): days
  - `bc` (float): km/s
  - `vrel` (float): km/s
  - `vrelerr` (float): km/s
  - `vhelio` (float): km/s

  _____

  Args:
    - visit_id: str

  Returns:
    - res. Has columns 'fiberid', 'plate', 'mjd', 'jd', 'bc', 'vrel', 'vrelerr', 'vhelio'
  """
  query = f"""
  SELECT fiberid, plate, mjd, jd, bc, vrel, vrelerr, vhelio
  FROM apogeeVisit
  WHERE visit_id = "{visit_id}"
  """
  res = SDSS.query_sql(query, data_release=19)
  return res

def _liststr(column):
  '''Converts column in astroquery `res` to list of strings'''
  lst = [str(x) for x in column]
  return lst

def _parseargs():
  '''
  Parses arguments
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--top', type=int, default=100)
  parser.add_argument('--visits', type=int, default=7)
  return parser.parse_args()

if __name__ == "__main__":
    '''
    Downlaod RV data from SDSS APOGEE database, saves as pickle file.

    Resulting file has the following structure:
    {
    `apstar_id_1` : {
        'jd' : np.array [1, 2, ...],
        'vhelio' : np.array [1, 2, ...]
        },
    `apstar_id_2` : {
        'jd' : np.array [1, 2, ...],
        'vhelio' : np.array [1, 2, ...]
        },
    ...
    }
    '''
    args = _parseargs()

    ### Construct an RV of a single system over multiple visits
    # Get apstar_id's with more than __ visits
    apstar_ids = _liststr(get_apstarId(visits=args.visits, top=args.top)['apstar_id'])

    # Construct {apstar_id : visit_id}
    visit_ids_per_apstar_id = {apstar_id: _liststr(get_visitIds_from_apstarId(apstar_id)['visit_id']) for apstar_id in tqdm(apstar_ids)}

    # Construct data container
    '''
    {
    apstar_id : {
        'jd' : np.array [1, 2, ...],
        'vhelio' : np.array [1, 2, ...]
    }
    }
    '''
    data = {}
    for apstar_id, visit_ids in tqdm(visit_ids_per_apstar_id.items()):
        # Get data from each visit
        jd = []
        vhelio = []
        vrelerr = []
        for visit_id in visit_ids:
            res = get_rvs_from_visit_id(visit_id)
            jd.append(res['jd'])
            vhelio.append(res['vhelio'])
            vrelerr.append(res['vrelerr']) # HACK: don't do this... Too hardcoded...

        # Convert to numpy arrays
        jd = np.array(jd).flatten()
        vhelio = np.array(vhelio).flatten()
        vrelerr = np.array(vrelerr).flatten()

        # Log data
        data[apstar_id] = {
            'jd' : jd,
            'vhelio' : vhelio,
            'vrelerr' : vrelerr
        }

    # Save data
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



