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
        self.F_matrix = np.eye(4)  # Placeholder for the state transition matrix
        self.Q_matrix = np.eye(4)  # Placeholder for the process noise covariance
        self.H_matrix = np.eye(4)  # Placeholder for the measurement matrix
        self.R_matrix = np.eye(4)  # Placeholder for the measurement noise covariance

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        # Define the state transition matrix F here
        dt = params.dt  # Time step
        self.F_matrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return self.F_matrix
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        # Define the process noise covariance matrix Q here
        dt = params.dt
        q = params.process_noise  # Process noise magnitude
        self.Q_matrix = np.array([
            [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
            [0, q * dt**4 / 4, 0, q * dt**3 / 2],
            [q * dt**3 / 2, 0, q * dt**2, 0],
            [0, q * dt**3 / 2, 0, q * dt**2]
        ])
        return self.Q_matrix
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F_matrix = self.F()
        Q_matrix = self.Q()

        # Prediction step
        track.x = F_matrix @ track.x  # Predicted state
        track.P = F_matrix @ track.P @ F_matrix.T + Q_matrix  # Predicted covariance
        
        pass
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        H_matrix = self.H()
        R_matrix = self.R()

        # Compute Kalman Gain
        S_matrix = self.S(track, meas, H_matrix)
        K = track.P @ H_matrix.T @ np.linalg.inv(S_matrix)
        
        # Update step
        gamma = self.gamma(track, meas)
        track.x = track.x + K @ gamma
        track.P = (np.eye(track.x.shape[0]) - K @ H_matrix) @ track.P
        
        track.update_attributes(meas)
    
        ############
        # END student code
        ############ 

    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        H_matrix = self.H()
        z = meas.z
        z_pred = H_matrix @ track.x
        gamma = z - z_pred
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        R_matrix = self.R()
        S = H @ track.P @ H.T + R_matrix
        return S
        
        ############
        # END student code
        ############
