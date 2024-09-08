import numpy as np
import os
import sys

# Add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params

class Sensor:
    '''Sensor class including measurement matrix'''
    
    def __init__(self, name, calib):
        self.name = name
        
        if name == 'lidar':
            self.dim_meas = 3
            self.sens_to_veh = np.matrix(np.identity(4))  # Transformation sensor to vehicle coordinates
            self.fov = [-np.pi / 2, np.pi / 2]  # Field of view in radians
            
        elif name == 'camera':
            self.dim_meas = 2
            self.sens_to_veh = np.matrix(calib.extrinsic.transform).reshape(4, 4)  # Transformation sensor to vehicle coordinates
            self.f_i = calib.intrinsic[0]  # Focal length i-coordinate
            self.f_j = calib.intrinsic[1]  # Focal length j-coordinate
            self.c_i = calib.intrinsic[2]  # Principal point i-coordinate
            self.c_j = calib.intrinsic[3]  # Principal point j-coordinate
            self.fov = [-0.35, 0.35]  # Field of view in radians (adjusted to remove inaccurate boundary region)
        
        # Transformation vehicle to sensor coordinates
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh)

    def in_fov(self, x):
        '''Check if an object x can be seen by this sensor'''
        
        if self.name == 'lidar':
            return True  # Lidar usually has a wider field of view
        
        elif self.name == 'camera':
            # Convert position x from vehicle to camera coordinates
            pos_veh = np.ones((4, 1))  # Homogeneous coordinates
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens @ pos_veh
            
            # Check if the position lies within the field of view
            x_cam, y_cam = pos_sens[0, 0], pos_sens[1, 0]
            if self.fov[0] <= np.arctan2(y_cam, x_cam) <= self.fov[1]:
                return True
            return False
        
        return False

    def get_hx(self, x):
        '''Calculate nonlinear measurement expectation value h(x)'''
        
        if self.name == 'lidar':
            pos_veh = np.ones((4, 1))  # Homogeneous coordinates
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens @ pos_veh  # Transform from vehicle to lidar coordinates
            return pos_sens[0:3]
        
        elif self.name == 'camera':
            # Transform position estimate from vehicle to camera coordinates
            pos_veh = np.ones((4, 1))  # Homogeneous coordinates
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens @ pos_veh
            
            # Projection from camera to image coordinates
            x_cam, y_cam = pos_sens[0, 0], pos_sens[1, 0]
            z_cam = pos_sens[2, 0]
            
            if z_cam == 0:
                raise ValueError("Division by zero in projection calculation.")
            
            u = self.f_i * x_cam / z_cam + self.c_i
            v = self.f_j * y_cam / z_cam + self.c_j
            
            return np.array([u, v])

    def get_H(self, x):
        '''Calculate Jacobian H at current x from h(x)'''
        
        H = np.matrix(np.zeros((self.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3]  # Rotation
        T = self.veh_to_sens[0:3, 3]  # Translation
        
        if self.name == 'lidar':
            H[0:3, 0:3] = R
        
        elif self.name == 'camera':
            denom = R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]
            
            if denom == 0:
                raise ValueError('Jacobian not defined for this x!')
            else:
                H[0, 0] = self.f_i * (-R[1, 0] / denom + R[0, 0] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1]) / denom**2)
                H[1, 0] = self.f_j * (-R[2, 0] / denom + R[0, 0] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2]) / denom**2)
                H[0, 1] = self.f_i * (-R[1, 1] / denom + R[0, 1] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1]) / denom**2)
                H[1, 1] = self.f_j * (-R[2, 1] / denom + R[0, 1] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2]) / denom**2)
                H[0, 2] = self.f_i * (-R[1, 2] / denom + R[0, 2] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1]) / denom**2)
                H[1, 2] = self.f_j * (-R[2, 2] / denom + R[0, 2] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2]) / denom**2)
        
        return H

    def generate_measurement(self, num_frame, z, meas_list):
        '''Generate new measurement from this sensor and add to measurement list'''
        
        if self.name in ['lidar', 'camera']:
            meas = Measurement(num_frame, z, self)
            meas_list.append(meas)
        
        return meas_list

###################

class Measurement:
    '''Measurement class including measurement values, covariance, timestamp, sensor'''
    
    def __init__(self, num_frame, z, sensor):
        '''Create measurement object'''
        
        self.t = (num_frame - 1) * params.dt  # Time
        
        if sensor.name == 'lidar':
            # Load lidar measurement noise parameters
            sigma_lidar_x = params.sigma_lidar_x
            sigma_lidar_y = params.sigma_lidar_y
            sigma_lidar_z = params.sigma_lidar_z
            
            self.z = np.zeros((sensor.dim_meas, 1))  # Measurement vector
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.z[2] = z[2]
            self.sensor = sensor  # Sensor that generated this measurement
            
            # Measurement noise covariance matrix
            self.R = np.matrix([[sigma_lidar_x ** 2, 0, 0],
                                [0, sigma_lidar_y ** 2, 0],
                                [0, 0, sigma_lidar_z ** 2]])
            
            # Additional parameters
            self.width = z[4]
            self.length = z[5]
            self.height = z[3]
            self.yaw = z[6]
        
        elif sensor.name == 'camera':
            # Load camera measurement noise parameters
            sigma_camera_u = params.sigma_camera_u
            sigma_camera_v = params.sigma_camera_v
            
            self.z = np.zeros((sensor.dim_meas, 1))  # Measurement vector
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.sensor = sensor  # Sensor that generated this measurement
            
            # Measurement noise covariance matrix
            self.R = np.matrix([[sigma_camera_u ** 2, 0],
                                [0, sigma_camera_v ** 2]])
        
        self.timestamp = self.t
