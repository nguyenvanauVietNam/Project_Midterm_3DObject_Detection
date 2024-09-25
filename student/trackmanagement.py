# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ---------------------------------------------------------------------
# https://github.com/udacity/nd013-c2-fusion-starter.git
# https://github.com/adamdivak/udacity_sd_lidar_fusion.git
# https://github.com/mabhi16/3D_Object_detection_midterm.git
# https://github.com/polarbeargo/nd013-Mid-Term-Project-3D-Object-Detection.git
#

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print('creating track no.', id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        ############
        # TODO Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on 
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        # self.x = np.matrix([[49.53980697],
        #                 [ 3.41006279],
        #                 [ 0.91790581],
        #                 [ 0.        ],
        #                 [ 0.        ],
        #                 [ 0.        ]])
        # self.P = np.matrix([[9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 6.4e-03, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+01]])
        position_sensor_frame = np.ones((4, 1))  # Create a column vector of ones for homogeneous coordinates
        position_sensor_frame[0:3] = meas.z[0:3]  # Assign the first three elements from the measurement's position

        position_vehicle_frame = meas.sensor.sens_to_veh * position_sensor_frame  # Transform from sensor frame to vehicle frame

        self.x = np.zeros((6, 1))  # Initialize state vector (position + velocity) with zeros
        self.x[0:3] = position_vehicle_frame[0:3]  # Assign transformed position (x, y, z) to the state vector
        #position_covariance = M_rot * meas.R * M_rot.T  # Rotate and optimize position covariance matrix
        position_covariance = M_rot * meas.R * np.transpose(M_rot) # Rotate and optimize position covariance matrix

        velocity_covariance = np.matrix([  # Optimize velocity covariance based on observed speed
            [params.sigma_p44**2, 0, 0], 
            [0, params.sigma_p55**2, 0], 
            [0, 0, params.sigma_p66**2]
        ])

        self.P = np.zeros((6, 6))  # Initialize covariance matrix for position and velocity
        self.P[0:3, 0:3] = position_covariance  # Assign optimized position covariance
        self.P[3:6, 3:6] = velocity_covariance  # Assign optimized velocity covariance

        self.state = 'initialized'  # Mark the track state as initialized
        self.score = 1/params.window #update param
        self.last_detections = collections.deque(params.window * [0], params.window)  # Initialize deque to track detection history
        self.last_detections.append(1)  # Append 1 to indicate a successful detection in the current frame

        self.score = sum(self.last_detections) / len(self.last_detections)  # Update the score based on the last detections

        ############
        # END student code
        ############ 
               
        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # Use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c * meas.width + (1 - c) * self.width
            self.length = c * meas.length + (1 - c) * self.length
            self.height = c * meas.height + (1 - c) * self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        
###################        

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############
        
        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility    
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    track.state =  'tentative' #update code base on memtor support
                    if track.score > params.delete_threshold + 1:  # Kiểm tra xem điểm số có lớn hơn ngưỡng xóa không
                            track.score = params.delete_threshold + 1  # Giới hạn điểm số không vượt quá ngưỡng xóa + 1
                    # your code goes here
                    #track.last_detections.append(0)  # Append a 0 to the detection history, indicating no detection for this frame
                    #track.score = sum(track.last_detections) / len(track.last_detections)  # Update the score as the average of last detections 
                    track.score -= 1. / params.window  # Giảm điểm số theo tỷ lệ 1/ kích thước cửa sổ, thể hiện sự suy giảm theo thời gian
        # delete old tracks 
        for track in self.track_list: # Loop through all tracks in the track list
            # #Check if the track is not initialized and has a low score
            # if track.state != 'initialized' and track.score < params.delete_threshold  \
            #     or track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P:
            #     self.delete_track(track) # Delete the track if any condition is met
            #Check if the track is not initialized and has a low score
            if track.score <= params.delete_threshold:
                if track.P[0, 0] >=  params.max_P or track.P[1, 1] >= params.max_P:
                    self.delete_track(track) # Delete the track if any condition is met
            
        ############
        # END student code
        ############ 
            
        # Initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        #track.last_detections.append(1)  # Append 1 to detection history (indicating a successful detection)
        
        #track.score = sum(track.last_detections) / len(track.last_detections)  # Update the tracking score based on the last detections
        track.score += 1./params.window  # Update the tracking score based on params.window
        # Set track state based on score: confirmed if score exceeds the threshold, tentative otherwise
        track.state = 'confirmed' if track.score > params.confirmed_threshold else 'tentative'

        
        ############
        # END student code
        ############ 
