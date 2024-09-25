# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
# https://github.com/udacity/nd013-c2-fusion-starter.git
# https://github.com/adamdivak/udacity_sd_lidar_fusion.git
# https://github.com/mabhi16/3D_Object_detection_midterm.git
# https://github.com/polarbeargo/nd013-Mid-Term-Project-3D-Object-Detection.git

# imports
import numpy as np
from scipy.stats.distributions import chi2

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
        self.association_matrix = np.matrix([])  # Ma trận kết hợp
        self.unassigned_tracks = []  # Danh sách các track chưa được gán
        self.unassigned_measurements = []  # Danh sách các đo lường chưa được gán

    def associate(self, track_list, meas_list, KF):
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # the following only works for at most one track and one measurement
        self.association_matrix = np.matrix([]) # reset matrix
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []
        
        if len(meas_list) > 0:
            self.unassigned_meas = [0]
        if len(track_list) > 0:
            self.unassigned_tracks = [0]
        if len(meas_list) > 0 and len(track_list) > 0: 
            self.association_matrix = np.matrix([[0]])
        
        # Initialize the number of tracks and measurements
        num_tracks = len(track_list)  # Number of tracks
        num_meas = len(meas_list)  # Number of measurements
        
        # Initialize unassigned tracks and measurements
        #self.association_matrix = np.ones((num_tracks, num_meas)) * np.inf # reset association_matrix
        self.association_matrix = np.asmatrix(np.inf * np.ones((num_tracks,num_meas)))# reset association_matrix
        #FIX bug by mentor Udacity
        #self.association_matrix  = list(range(num_meas))  # Measurement indices
        self.unassigned_meas = list(range(num_meas))        # Measurement indices
        self.unassigned_tracks = list(range(num_tracks))  # Track indices
        
        # Create an association matrix filled with infinity
        #self.association_matrix = np.asmatrix(np.inf * np.ones((num_tracks, num_meas)))  # Association matrix

        # Iterate through tracks and measurements to calculate distances
        # for track_index in range(num_tracks):
        #     track = track_list[track_index]  # Current track
        #     for meas_index in range(num_meas):
        #         meas = meas_list[meas_index]  # Current measurement
        #Fix by mentor Udacity
        for track_index in range(num_tracks): #fix bug TypeError: list indices must be integers or slices, not tuple
            track = track_list[track_index] #Fix bug bug QnA: https://knowledge.udacity.com/questions/1051580
            for meas_index in range(num_meas):        
                try:
                    # dist = self.MHD(track, meas, KF)  # Calculate Mahalanobis distance
                    # # Check if the distance is within the gating criteria
                    # if self.gating(dist, meas.sensor):
                    #     self.association_matrix[track_index, meas_index] = dist  # Update association matrix with distance
                    #Fix bug QnA: https://knowledge.udacity.com/questions/1051580
                    meas = meas_list[meas_index] #Fix bug bug QnA: https://knowledge.udacity.com/questions/1051580
                    dist = self.MHD(track, meas, KF)  # Calculate Mahalanobis distance
                    if self.gating(dist, meas.sensor):
                        self.association_matrix[track_index, meas_index] = dist
                except Exception as e:
                    print(f"Error calculating distance for track {track_index} and measurement {meas_index}: {e}")
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
        # the following only works for at most one track and one measurement
        update_track = 0
        update_meas  = 0

        # Check for minimum entry in the association matrix
        association_matrix = self.association_matrix
        if np.min(association_matrix) == np.inf:
            return np.nan, np.nan  # Return NaN if no valid associations exist

        # Get indices of the minimum entry in the association matrix
        min_indices = np.unravel_index(np.argmin(association_matrix, axis=None), association_matrix.shape) 
        indices_track = min_indices[0]
        indices_meas = min_indices[1]

        # Delete the corresponding row and column for the next update
        association_matrix = np.delete(association_matrix, indices_track, 0) 
        association_matrix = np.delete(association_matrix, indices_meas, 1)
        self.association_matrix = association_matrix

        # Update the closest track and measurement using the indices
        update_track = self.unassigned_tracks[indices_track] 
        update_meas = self.unassigned_meas[indices_meas]

        # remove from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        #self.association_matrix = np.matrix([])

        ############
        # END student code
        ############ 
        # Return the closest track and measurement indices
        return update_track, update_meas   




    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
            # gating_threshold_value: the threshold value based on the chi-squared distribution that determines if the measurement is inside the gate
            gating_threshold_value = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
            
            # return True if MHD < gating_threshold_value else return FALSE
            return MHD < gating_threshold_value
        ############
        # END student code
        ############ 
        
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
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

    def MHD(self, track, meas, KF):
            ############
            # TODO Step 3: calculate and return Mahalanobis distance
            ############
            
            #S = KF.S(track, meas, meas.sensor.get_H(track.x))  # Calculate the innovation covariance matrix S
            S = np.linalg.inv(KF.S(track, meas, meas.sensor.get_H(track.x)))  # Tính ma trận S, sau đó lấy nghịch đảo của nó
            mhd_gamma = KF.gamma(track, meas)  # Get gamma for KF

            mhd_result = mhd_gamma.T * S * mhd_gamma  # Compute the Mahalanobis distance result using the transpose of gamma and the inverse of S
            
            return mhd_result  # Return the final Mahalanobis distance result
            ############
            # END student code
            ############ 