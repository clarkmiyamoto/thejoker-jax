from astropy.io import fits
import numpy as np

class APOGEE_DATA:
    '''
    Helps parsing APOGEE data
    '''

    def __init__(self, filename: str):
        self.allVisit = fits.open(filename)
        self.data = self.allVisit[1].data

    def getRadialVelocity(self, apogee_id: str):
        '''
        Get radial velocity and error for a given apogee_id

        Args:
            - apogee_id

        Returns:
            - times
            - radial_velocity
            - radial_velocity_error
        '''
        data = self.data

        visit_id = np.where(data['apogee_id'] == apogee_id)[0]
        
        times = data['MJD'][visit_id]
        radial_velocity =  data['VREL'][visit_id]
        radial_velocity_error = data['VRELERR'][visit_id]
        return times, radial_velocity, radial_velocity_error