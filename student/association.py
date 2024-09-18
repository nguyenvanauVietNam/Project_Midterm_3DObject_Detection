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

    def associate(self, track_list, measurement_list, kalman_filter):
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # Reset association matrix and lists for tracks and measurements
        self.association_matrix = np.matrix([])  # Reset ma trận kết hợp
        self.unassigned_tracks = []  # Reset danh sách track chưa được gán
        self.unassigned_measurements = []  # Reset danh sách đo lường chưa được gán
        
        # Initialize lists with indices if there are any tracks or measurements
        if len(measurement_list) > 0:
            self.unassigned_measurements = list(range(len(measurement_list)))
        if len(track_list) > 0:
            self.unassigned_tracks = list(range(len(track_list)))
        
        # Initialize the association matrix with infinities
        num_tracks = len(track_list)
        num_measurements = len(measurement_list)
        self.association_matrix = np.asmatrix(np.inf * np.ones((num_tracks, num_measurements)))

        # Compute Mahalanobis distance for each track and measurement pair
        for track_index in range(num_tracks):
            track = track_list[track_index]
            for measurement_index in range(num_measurements):
                measurement = measurement_list[measurement_index]
                mahalanobis_distance = self.calculate_mahalanobis_distance(track, measurement, kalman_filter)
                if self.is_within_gating_distance(mahalanobis_distance, measurement.sensor):
                    self.association_matrix[track_index, measurement_index] = mahalanobis_distance

    def get_closest_track_and_measurement(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_measurements
        # - return this track and measurement
        ############

        # Check if there are any valid associations
        if np.min(self.association_matrix) == np.inf:
            return np.nan, np.nan

        # Find the index of the minimum entry in the association matrix
        min_index = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        track_index = min_index[0]
        measurement_index = min_index[1]

        # Get the corresponding track and measurement
        closest_track = self.unassigned_tracks[track_index]
        closest_measurement = self.unassigned_measurements[measurement_index]

        # Remove the assigned track and measurement from the lists
        self.unassigned_tracks.remove(closest_track)
        self.unassigned_measurements.remove(closest_measurement)

        # Delete the corresponding row and column in the association matrix
        self.association_matrix = np.delete(self.association_matrix, track_index, 0)
        self.association_matrix = np.delete(self.association_matrix, measurement_index, 1)

        return closest_track, closest_measurement

    def is_within_gating_distance(self, mahalanobis_distance, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        gating_limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        return mahalanobis_distance < gating_limit

    def calculate_mahalanobis_distance(self, track, measurement, kalman_filter):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        measurement_matrix = measurement.sensor.get_H(track.x)
        measurement_residual = kalman_filter.gamma(track, measurement)
        covariance_inverse = np.linalg.inv(kalman_filter.S(track, measurement, measurement_matrix))
        mahalanobis_distance = measurement_residual.T * covariance_inverse * measurement_residual
        return mahalanobis_distance

    def associate_and_update(self, track_manager, measurement_list, kalman_filter):
        # Associate measurements and tracks
        self.associate(track_manager.track_list, measurement_list, kalman_filter)

        # Update associated tracks with measurements
        while self.association_matrix.shape[0] > 0 and self.association_matrix.shape[1] > 0:

            # Search for the next association between a track and a measurement
            track_index, measurement_index = self.get_closest_track_and_measurement()
            if np.isnan(track_index):
                print('---no more associations---')
                break
            track = track_manager.track_list[track_index]

            # Check visibility; only update tracks in field of view
            if not measurement_list[0].sensor.in_fov(track.x):
                continue

            # Perform Kalman update
            print('Update track', track.id, 'with', measurement_list[measurement_index].sensor.name, 'measurement', measurement_index)
            kalman_filter.update(track, measurement_list[measurement_index])

            # Update score and track state
            track_manager.handle_updated_track(track)

            # Save updated track
            track_manager.track_list[track_index] = track

        # Run track management
        track_manager.manage_tracks(self.unassigned_tracks, self.unassigned_measurements, measurement_list)

        # Print the score of each track
        for track in track_manager.track_list:            
            print('Track', track.id, 'score =', track.score)
