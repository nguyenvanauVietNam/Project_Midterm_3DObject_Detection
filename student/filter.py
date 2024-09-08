import numpy as np

# Add project directory to Python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.f_matrix = np.eye(4)  # Placeholder for the state transition matrix
        self.q_matrix = np.eye(4)  # Placeholder for the process noise covariance
        self.h_matrix = np.eye(4)  # Placeholder for the measurement matrix
        self.r_matrix = np.eye(4)  # Placeholder for the measurement noise covariance

    def f(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        # Define the state transition matrix F here
        dt = params.dt  # Time step
        self.f_matrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return self.f_matrix
        
        ############
        # END student code
        ############ 

    def q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        # Define the process noise covariance matrix Q here
        dt = params.dt
        q = params.process_noise  # Process noise magnitude
        self.q_matrix = np.array([
            [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
            [0, q * dt**4 / 4, 0, q * dt**3 / 2],
            [q * dt**3 / 2, 0, q * dt**2, 0],
            [0, q * dt**3 / 2, 0, q * dt**2]
        ])
        return self.q_matrix
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        f_matrix = self.f()
        q_matrix = self.q()

        # Prediction step
        track.x = f_matrix @ track.x  # Predicted state
        track.P = f_matrix @ track.P @ f_matrix.T + q_matrix  # Predicted covariance
        
        pass
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        h_matrix = self.h()
        r_matrix = self.r()

        # Compute Kalman Gain
        s_matrix = self.s(track, meas, h_matrix)
        k_gain = track.P @ h_matrix.T @ np.linalg.inv(s_matrix)
        
        # Update step
        gamma = self.gamma(track, meas)
        track.x = track.x + k_gain @ gamma
        track.P = (np.eye(track.x.shape[0]) - k_gain @ h_matrix) @ track.P
        
        track.update_attributes(meas)
    
        ############
        # END student code
        ############ 

    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        h_matrix = self.h()
        z = meas.z
        z_pred = h_matrix @ track.x
        gamma = z - z_pred
        return gamma
        
        ############
        # END student code
        ############ 

    def s(self, track, meas, h_matrix):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        r_matrix = self.r()
        s_matrix = h_matrix @ track.P @ h_matrix.T + r_matrix
        return s_matrix
        
        ############
        # END student code
        ############
