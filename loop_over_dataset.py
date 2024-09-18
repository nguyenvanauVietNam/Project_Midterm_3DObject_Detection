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

## general package imports
import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

## 3d object detection
import student.objdet_pcl as pcl
import student.objdet_detect as det
import student.objdet_eval as eval

import misc.objdet_tools as tools 
from misc.helpers import save_object_to_file, load_object_from_file, make_exec_list

from student.filter import Filter
from student.trackmanagement import Trackmanagement
from student.association import Association
from student.measurements import Sensor, Measurement
from misc.evaluation import plot_tracks, plot_rmse, make_movie
import misc.params as params 

##################
## Set parameters and perform initializations

#Section 1 : Compute Lidar Point-Cloud from Range Image###
##Visualize range image channels (ID_S1_EX1)  
## Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
show_only_frames = [0, 1] # show only frames in interval for debugging

## Prepare Waymo Open Dataset file for loading
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename)
model_name = "darknet"
sequence_num = "1"
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model_name + '/results_sequence_' + sequence_num + '_' + model_name)
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # initialize dataset iterator

## Initialize object detection
configs_det = det.load_configs(model_name='darknet') # options are 'darknet', 'fpn_resnet'
model_det = det.create_model(configs_det)

configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

## Selective execution and visualization
exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
exec_tracking = [] 
exec_visualization = ['show_range_image'] 
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 

## Set parameters and perform initializations

# ## Select Waymo Open Dataset file and frame numbers
# # data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# # data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
# data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 3
# show_only_frames = [0, 200] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "3"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name='darknet') # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Initialize tracking
# KF = Filter() # set up Kalman filter 
# association = Association() # init data association
# manager = Trackmanagement() # init track manager
# lidar = None # init lidar sensor object
# camera = None # init camera sensor object
# np.random.seed(10) # make random values predictable

# ## Selective execution and visualization
# exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)

# ##Visualize lidar point-cloud (ID_S1_EX2)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [0, 200] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name='darknet') # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = []
# exec_detection = []
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = ['show_pcl'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)

# #Section 2 : Create Birds-Eye View from Lidar PCL
# ###Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [0, 1] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name='darknet') # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = ['pcl_from_rangeimage']
# exec_detection = ['bev_from_pcl']
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)


# ##Compute intensity layer of the BEV map (ID_S2_EX2)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [0, 1] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name='darknet') # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = ['pcl_from_rangeimage']
# exec_detection = ['bev_from_pcl']
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)

# ##Compute height layer of the BEV map (ID_S2_EX3)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [0, 1] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name='darknet') # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = ['pcl_from_rangeimage']
# exec_detection = ['bev_from_pcl']
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)

# #Section 3 : Model-based Object Detection in BEV Image
# ##Add a second model from a GitHub repo (ID_S3_EX1)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [50, 51] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name="fpn_resnet") # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = ['pcl_from_rangeimage', 'load_image']
# exec_detection = ['bev_from_pcl', 'detect_objects']
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = ['show_objects_in_bev_labels_in_camera'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)


# ##Extract 3D bounding boxes from model response (ID_S3_EX2)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [50, 51] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name="fpn_resnet") # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = ['pcl_from_rangeimage', 'load_image']
# exec_detection = ['bev_from_pcl', 'detect_objects']
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = ['show_objects_in_bev_labels_in_camera'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)

# #Section 4 : Performance Evaluation for Object Detection
# ##Compute intersection-over-union between labels and detections (ID_S4_EX1)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [50, 51] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name="darknet") # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = ['pcl_from_rangeimage']
# exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = ['show_detection_performance'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)

# ##Compute precision and recall (ID_S4_EX3)
# ## Select Waymo Open Dataset file and frame numbers
# data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# show_only_frames = [50, 150] # show only frames in interval for debugging

# ## Prepare Waymo Open Dataset file for loading
# data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
# model = "darknet"
# sequence = "1"
# results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
# datafile = WaymoDataFileReader(data_fullpath)
# datafile_iter = iter(datafile)  # initialize dataset iterator

# ## Initialize object detection
# configs_det = det.load_configs(model_name="darknet") # options are 'darknet', 'fpn_resnet'
# model_det = det.create_model(configs_det)

# configs_det.use_labels_as_objects = True # True = use groundtruth labels as objects, False = use model-based detection

# ## Uncomment this setting to restrict the y-range in the final project
# # configs_det.lim_y = [-25, 25] 

# ## Selective execution and visualization
# exec_data = ['pcl_from_rangeimage']
# exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
# exec_tracking = [] # options are 'perform_tracking'
# exec_visualization = ['show_detection_performance'] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
# vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)

##################
## Initialize tracking
kalman_filter = Filter() # set up Kalman filter 
association = Association() # init data association
track_manager = Trackmanagement() # init track manager
lidar_sensor = None # init lidar sensor object
camera_sensor = None # init camera sensor object
np.random.seed(10) # make random values predictable

##################
## Perform detection & tracking over all selected frames

frame_count = 0 
all_labels = []
detection_performance_all = [] 
if 'show_tracks' in exec_list:    
    fig, (ax2, ax) = plt.subplots(1,2) # init track plot

while True:
    try:
        ## Get next frame from Waymo dataset
        frame = next(datafile_iter)
        if frame_count < show_only_frames[0]:
            frame_count += 1
            continue
        elif frame_count > show_only_frames[1]:
            print('Reached end of selected frames')
            break
        
        print('------------------------------')
        print(f'Processing frame #{frame_count}')

        #################################
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from frame
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)        
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
        if 'load_image' in exec_list:
            image = tools.extract_front_camera_image(frame) 

        ## Compute lidar point-cloud from range image    
        if 'pcl_from_rangeimage' in exec_list:
            print('Computing point-cloud from lidar range image')
            lidar_pcl = tools.pcl_from_range_image(frame, lidar_name)
        else:
            print('Loading lidar point-cloud from result file')
            lidar_pcl = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', frame_count)
            
        ## Compute lidar birds-eye view (bev)
        if 'bev_from_pcl' in exec_list:
            print('Computing birds-eye view from lidar pointcloud')
            lidar_bev = pcl.bev_from_pcl(lidar_pcl, configs_det)
        else:
            print('Loading birds-eye view from result file')
            lidar_bev = load_object_from_file(results_fullpath, data_filename, 'lidar_bev', frame_count)

        ## 3D object detection
        if configs_det.use_labels_as_objects:
            print('Using groundtruth labels as objects')
            detections = tools.convert_labels_into_objects(frame.laser_labels, configs_det)
        else:
            if 'detect_objects' in exec_list:
                print('Detecting objects in lidar pointcloud')   
                detections = det.detect_objects(lidar_bev, model_det, configs_det)
            else:
                print('Loading detected objects from result file')
                detections = load_object_from_file(results_fullpath, data_filename, f'detections_{configs_det.arch}_{configs_det.conf_thresh}', frame_count)

        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("Validating object labels")
            valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects else 10)
        else:
            print('Loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(results_fullpath, data_filename, 'valid_labels', frame_count)            

        ## Performance evaluation for object detection
        if 'measure_detection_performance' in exec_list:
            print('Measuring detection performance')
            detection_performance = eval.measure_detection_performance(detections, frame.laser_labels, valid_label_flags, configs_det.min_iou)     
        else:
            print('Loading detection performance measures from file')
            detection_performance = load_object_from_file(results_fullpath, data_filename, f'det_performance_{configs_det.arch}_{configs_det.conf_thresh}', frame_count)   

        detection_performance_all.append(detection_performance)

        ## Visualization for object detection
        if 'show_range_image' in exec_list:
            img_range = pcl.show_range_image(frame, lidar_name)
            img_range = img_range.astype(np.uint8)
            cv2.imshow('range_image', img_range)
            cv2.waitKey(vis_pause_time)

        if 'show_pcl' in exec_list:
            pcl.show_pcl(lidar_pcl)

        if 'show_bev' in exec_list:
            tools.show_bev(lidar_bev, configs_det)  
            cv2.waitKey(vis_pause_time)

        if 'show_labels_in_image' in exec_list:
            img_labels = tools.project_labels_into_camera(camera_calibration, image, frame.laser_labels, valid_label_flags, 0.5)
            cv2.imshow('img_labels', img_labels)
            cv2.waitKey(vis_pause_time)

        if 'show_objects_and_labels_in_bev' in exec_list:
            tools.show_objects_labels_in_bev(detections, frame.laser_labels, lidar_bev, configs_det)
            cv2.waitKey(vis_pause_time)        

        if 'show_objects_in_bev_labels_in_camera' in exec_list:
            tools.show_objects_in_bev_labels_in_camera(detections, lidar_bev, image, frame.laser_labels, valid_label_flags, camera_calibration, configs_det)
            cv2.waitKey(vis_pause_time)               

        #################################
        ## Perform tracking
        if 'perform_tracking' in exec_list:
            # set up sensor objects
            if lidar_sensor is None:
                lidar_sensor = Sensor('lidar', lidar_calibration)
            if camera_sensor is None:
                camera_sensor = Sensor('camera', camera_calibration)
            
            # preprocess lidar detections
            lidar_meas_list = []
            for detection in detections:
                lidar_meas_list = lidar_sensor.generate_measurement(frame_count, detection[1:], lidar_meas_list)
                
            # preprocess camera detections
            camera_meas_list = []
            for label in frame.camera_labels[0].labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    box = label.box
                    z = [box.center_x, box.center_y, box.width, box.length]
                    z[0] += np.random.normal(0, params.sigma_cam_i)  # add noise
                    z[1] += np.random.normal(0, params.sigma_cam_i) 
                    camera_meas_list = camera_sensor.generate_measurement(frame_count, z, camera_meas_list)

            # run Kalman filter for lidar and camera measurements
            kalman_filter.predict()
            track_manager.manage_tracks(frame_count)
            association.associate_and_update(frame_count, kalman_filter, track_manager, lidar_meas_list, camera_meas_list)

        ## Visualization for tracking
        if 'show_tracks' in exec_list:
            ax.cla()
            ax2.cla()
            plot_tracks(ax, ax2, track_manager, frame_count)

        ## Increment frame counter
        frame_count += 1  

    except StopIteration:
        ## Stop if end of file has been reached
        print("End of dataset")
        break

#################################
## Post-processing

## Plot RMSE for all tracks
if 'show_tracks' in exec_list:
    plot_rmse(track_manager)
    make_movie(data_filename)

## Store all evaluation results in a file
print('Storing results')
eval.store_results(data_filename, detection_performance_all, exec_list, results_fullpath)
