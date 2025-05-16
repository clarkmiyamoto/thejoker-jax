import sdss_access
import numpy as np
import matplotlib.pyplot as plt


config = {
    'release': 'sdsswork',
    'telescope': 'apo25m',
    'apred': '1.4'
}

if __name__ == '__main__':
    sdss_path = sdss_access.path.Path(release=config['release'], verbose=True)
    sdss_http = sdss_access.HttpAccess(release=config['release'], verbose=True)
    sdss_http.remote()
    sdss_http.get('allVisit', telescope=config['telescope'], apred=config['apred'])
    filename = sdss_http.full('allVisit', telescope=config['telescope'], apred=config['apred'])