# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Loop over all frames in a Waymo Open Dataset file,
#                        detect and track objects and visualize results
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

##################
## Imports

# Import general libraries
import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy

# Add the current working directory to the path to import modules from it
sys.path.append(os.getcwd())

# Import Waymo Open Dataset utilities for reading data
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

# Import 3D object detection and tracking related modules
import student.objdet_pcl as pcl
import student.objdet_detect as det
import student.objdet_eval as eval

# Import additional tools and utility functions
import misc.objdet_tools as tools 
from misc.helpers import save_object_to_file, load_object_from_file, make_exec_list

# Import Kalman filter, tracking management, association, and measurement classes
from student.filter import Filter
from student.trackmanagement import Trackmanagement
from student.association import Association
from student.measurements import Sensor, Measurement
from misc.evaluation import plot_tracks, plot_rmse, make_movie
import misc.params as params 
 
##################
## Set parameters and perform initializations

# Specify the Waymo Open Dataset file and frame range for visualization
data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Choose the data file
show_only_frames = [0, 200] # Only show frames within this range for debugging

# Prepare the full path to the data file
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) 
model = "darknet" # Select the detection model
sequence = "3" # Choose the sequence
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model) # Path to results
datafile = WaymoDataFileReader(data_fullpath) # Read the Waymo data file
datafile_iter = iter(datafile)  # Initialize iterator for the data

# Initialize object detection
configs_det = det.load_configs(model_name='darknet') # Load model configuration
model_det = det.create_model(configs_det) # Create the object detection model

configs_det.use_labels_as_objects = False # Use model-based detection (not ground truth labels)

## Uncomment this setting to restrict the y-range in the final project
# configs_det.lim_y = [-25, 25] 

# Initialize tracking
KF = Filter() # Initialize Kalman filter
association = Association() # Initialize data association
manager = Trackmanagement() # Initialize track management
lidar = None # Initialize lidar sensor object
camera = None # Initialize camera sensor object
np.random.seed(10) # Set random seed for reproducibility

# Set up the execution list for filtering and visualization
exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'] # Detection tasks
exec_tracking = [] # Tracking tasks
exec_visualization = [] # Visualization tasks
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization) # Create the list of tasks to be executed
vis_pause_time = 0 # Pause time between frames (0 = pause until key press)

##################
## Perform detection & tracking over all selected frames

cnt_frame = 0 # Frame counter
all_labels = [] # List to store all labels
det_performance_all = [] # List to store detection performance metrics
if 'show_tracks' in exec_list:    
    fig, (ax2, ax) = plt.subplots(1,2) # Initialize the tracking plot

while True:
    try:
        ## Get the next frame from the Waymo dataset
        frame = next(datafile_iter)
        if cnt_frame < show_only_frames[0]:
            cnt_frame = cnt_frame + 1
            continue
        elif cnt_frame > show_only_frames[1]:
            print('reached end of selected frames')
            break
        
        print('------------------------------')
        print('processing frame #' + str(cnt_frame))

        #################################
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from the frame
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)        
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
        if 'load_image' in exec_list:
            image = tools.extract_front_camera_image(frame) # Extract front camera image if needed

        ## Compute point cloud from lidar range image
        if 'pcl_from_rangeimage' in exec_list:
            print('computing point-cloud from lidar range image')
            lidar_pcl = tools.pcl_from_range_image(frame, lidar_name) # Compute point cloud from lidar range image
        else:
            print('loading lidar point-cloud from result file')
            lidar_pcl = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', cnt_frame) # Load point cloud from result file
            
        ## Compute birds-eye view (BEV) from lidar
        if 'bev_from_pcl' in exec_list:
            print('computing birds-eye view from lidar pointcloud')
            lidar_bev = pcl.bev_from_pcl(lidar_pcl, configs_det) # Compute BEV from lidar point cloud
        else:
            print('loading birds-eye view from result file')
            lidar_bev = load_object_from_file(results_fullpath, data_filename, 'lidar_bev', cnt_frame) # Load BEV from result file

        ## Detect 3D objects
        if (configs_det.use_labels_as_objects==True):
            print('using groundtruth labels as objects')
            detections = tools.convert_labels_into_objects(frame.laser_labels, configs_det) # Use ground truth labels to detect objects
        else:
            if 'detect_objects' in exec_list:
                print('detecting objects in lidar pointcloud')   
                detections = det.detect_objects(lidar_bev, model_det, configs_det) # Detect objects in lidar point cloud
            else:
                print('loading detected objects from result file')
                detections = load_object_from_file(results_fullpath, data_filename, 'detections_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame) # Load detected objects from result file

        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("validating object labels")
            valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects==True else 10) # Validate object labels
        else:
            print('loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(results_fullpath, data_filename, 'valid_labels', cnt_frame) # Load object labels and validation from result file            

        ## Measure detection performance
        if 'measure_detection_performance' in exec_list:
            print('measuring detection performance')
            det_performance = eval.measure_detection_performance(detections, frame.laser_labels, valid_label_flags, configs_det.min_iou) # Measure detection performance
        else:
            print('loading detection performance measures from file')
            det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame) # Load detection performance metrics from result file   

        det_performance_all.append(det_performance) # Append detection performance results to the list
        
        ## Visualize results
        if 'show_tracks' in exec_list:
            ax.clear()                
            # Show front camera image
            if 'load_image' in exec_list:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert color and display front camera image
                ax.set_title('Camera image') # Title for camera image plot

            # Show lidar birds-eye view
            ax2.clear() 
            ax2.imshow(np.max(lidar_bev, axis=2), cmap='gray') # Display BEV from lidar_bev
            ax2.set_title('Lidar BEV') # Title for lidar BEV plot

            plt.pause(vis_pause_time) # Pause between frames

        ## Perform object tracking
        if 'perform_tracking' in exec_list:
            print('performing tracking')
            for det in detections:
                # Convert detections to measurement format and track
                measurement = Measurement(det, lidar_pcl)
                association.match_measurement_with_tracks(measurement)
                manager.update_tracks(measurement)
            manager.cleanup_tracks()

        # Increment frame counter
        cnt_frame = cnt_frame + 1

    except StopIteration:
        print('finished processing all frames')
        break

##################
## Save results

# Save results
if 'show_tracks' in exec_list:
    plot_tracks(results_fullpath, data_filename, 'tracks_' + str(cnt_frame), exec_list) # Save tracking results
if 'measure_detection_performance' in exec_list:
    plot_rmse(det_performance_all, results_fullpath, data_filename, 'detection_performance_' + str(cnt_frame)) # Save detection performance results
if 'make_movie' in exec_list:
    make_movie(results_fullpath, data_filename, 'movie_' + str(cnt_frame)) # Create a video from results
