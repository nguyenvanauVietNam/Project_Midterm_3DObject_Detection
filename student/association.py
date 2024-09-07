import numpy as np
from scipy.stats.distributions import chi2
from scipy.linalg import inv

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        num_tracks = len(track_list)
        num_meas = len(meas_list)
        
        self.association_matrix = np.matrix(np.inf * np.ones((num_tracks, num_meas)))
        self.unassigned_tracks = list(range(num_tracks))
        self.unassigned_meas = list(range(num_meas))
        
        for i, track in enumerate(track_list):
            for j, meas in enumerate(meas_list):
                MHD = self.MHD(track, meas, KF)
                if self.gating(MHD, meas.sensor):
                    self.association_matrix[i, j] = MHD
        
        if num_tracks == 0 or num_meas == 0:
            self.association_matrix = np.matrix([])
            self.unassigned_tracks = []
            self.unassigned_meas = []

        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        if self.association_matrix.size == 0:
            return np.nan, np.nan

        i, j = np.unravel_index(np.argmin(self.association_matrix), self.association_matrix.shape)
        
        update_track = self.unassigned_tracks[i]
        update_meas = self.unassigned_meas[j]

        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)

        self.association_matrix = np.delete(self.association_matrix, i, axis=0)
        self.association_matrix = np.delete(self.association_matrix, j, axis=1)
        
        return update_track, update_meas

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        gating_threshold = chi2.ppf(0.95, df=2)
        return MHD <= gating_threshold
    
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        z = meas.z
        H = KF.H(track)
        P = KF.P(track)
        R = KF.R(meas)
        
        # Residual
        z_pred = KF.predict(track)
        residual = z - z_pred
        
        # Mahalanobis Distance
        S = H @ P @ H.T + R
        inv_S = inv(S)
        MHD = residual.T @ inv_S @ residual
        
        return MHD
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0] > 0 and self.association_matrix.shape[1] > 0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[ind_meas].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)
