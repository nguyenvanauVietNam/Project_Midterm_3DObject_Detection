# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: Implement and return system matrix F
        # The system matrix F describes the state transition. It should be implemented based on your specific model.
        # This matrix accounts for how the state evolves over time.
        ############
        delta_time = params.dt  # Time step from params
        return np.matrix([[1, 0, 0, delta_time, 0 ,0],
                          [0, 1, 0, 0, delta_time, 0],
                          [0, 0, 1, 0, 0, delta_time],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0], 
                          [0, 0, 0, 0, 0, 1]])
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: Implement and return process noise covariance Q
        # The process noise covariance matrix Q accounts for the uncertainty in the process model.
        # It defines how the noise affects the state over time.
        ############
        process_noise = params.q  # Process noise magnitude from params
        delta_time = params.dt  # Time step from params
        q1 = ((delta_time**3)/3) * process_noise 
        q2 = ((delta_time**2)/2) * process_noise 
        q3 = delta_time * process_noise 
        return np.matrix([[q1, 0, 0, q2, 0, 0],
                          [0, q1, 0, 0, q2, 0],
                          [0, 0, q1, 0, 0, q2],
                          [q2, 0, 0, q3, 0, 0],
                          [0, q2, 0, 0, q3, 0],
                          [0, 0, q2, 0, 0, q3]])
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: Predict state x and estimation error covariance P to the next timestep
        # This method should update the state and covariance estimates for the track based on the Kalman filter equations.
        # It performs the prediction step using the system matrix F and process noise covariance Q.
        ############
        system_matrix = self.F()  # System matrix F
        state_estimate = track.x  # Current state estimate
        error_covariance = track.P  # Current estimation error covariance
        
        # Predict the state and covariance
        state_prediction = system_matrix @ state_estimate
        covariance_prediction = system_matrix @ error_covariance @ system_matrix.T + self.Q()
        
        # Update track with predictions
        track.set_x(state_prediction)
        track.set_P(covariance_prediction)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: Update state x and covariance P with associated measurement
        # This method should adjust the state and covariance estimates based on the new measurement.
        # It performs the update step using the measurement matrix H, residual gamma, and covariance of residual S.
        ############
        measurement_matrix = meas.sensor.get_H(track.x)  # Measurement matrix H
        residual_gamma = self.gamma(track, meas)  # Residual gamma
        residual_covariance = self.S(track, meas, measurement_matrix)  # Covariance of residual S
        
        # Calculate Kalman gain
        kalman_gain = track.P @ measurement_matrix.T @ np.linalg.inv(residual_covariance)
        
        # Update state and covariance
        updated_state = track.x + kalman_gain @ residual_gamma
        identity_matrix = np.eye(params.dim_state)
        updated_covariance = (identity_matrix - kalman_gain @ measurement_matrix) @ track.P
        
        # Save updated state and covariance in track
        track.set_x(updated_state)
        track.set_P(updated_covariance)
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: Calculate and return residual gamma
        # Residual gamma is the difference between the actual measurement and the predicted measurement.
        # It measures how much the prediction deviates from the actual measurement.
        ############
        predicted_measurement = meas.sensor.get_hx(track.x)  # Predicted measurement
        residual_gamma = meas.z - predicted_measurement  # Measurement residual
        
        return residual_gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, measurement_matrix):
        ############
        # TODO Step 1: Calculate and return covariance of residual S
        # The covariance of the residual S is used to measure the uncertainty of the prediction.
        # It accounts for the noise in the measurement and the prediction.
        ############
        residual_covariance = measurement_matrix @ track.P @ measurement_matrix.T + meas.R  # Covariance of residual
        
        return residual_covariance
        
        ############
        # END student code
        ############ 
