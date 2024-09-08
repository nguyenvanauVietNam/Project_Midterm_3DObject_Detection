# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file: Loop over all frames in a Waymo Open Dataset file,
#                        detect and track objects and visualize results
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------

##################
## Imports

## General package imports
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo Open Dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

## 3D object detection
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

## Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'# Sequence 1
# data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
# data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 3
show_only_frames = [0, 200]  # Show only frames in interval for debugging

## Prepare Waymo Open Dataset file for loading
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename)
model_name = "darknet"
sequence_number = "3"
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'results/{model_name}/results_sequence_{sequence_number}_{model_name}')
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # Initialize dataset iterator

## Initialize object detection
det_config = det.load_configs(model_name=model_name)  # Options are 'darknet', 'fpn_resnet'
det_model = det.create_model(det_config)

det_config.use_labels_as_objects = False  # True = use groundtruth labels as objects, False = use model-based detection

## Uncomment this setting to restrict the y-range in the final project
# det_config.lim_y = [-25, 25] 

## Initialize tracking
kalman_filter = Filter()  # Set up Kalman filter 
data_association = Association()  # Init data association
track_manager = Trackmanagement()  # Init track manager
lidar_sensor = None  # Init lidar sensor object
camera_sensor = None  # Init camera sensor object
np.random.seed(10)  # Make random values predictable

## Selective execution and visualization
detection_tasks = ['pcl_from_rangeimage', 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
tracking_tasks = ['perform_tracking']
visualization_tasks = ['show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie']
exec_list = make_exec_list(detection_tasks, tracking_tasks, visualization_tasks)
visualization_pause_time = 0  # Set pause time between frames in ms (0 = stop between frames until key is pressed)

##################
## Perform detection & tracking over all selected frames

frame_counter = 0
all_labels = []
detection_performance_all = []
if 'show_tracks' in exec_list:
    fig, (ax2, ax) = plt.subplots(1, 2)  # Init track plot

while True:
    try:
        ## Get next frame from Waymo dataset
        frame = next(datafile_iter)
        if frame_counter < show_only_frames[0]:
            frame_counter += 1
            continue
        elif frame_counter > show_only_frames[1]:
            print('Reached end of selected frames')
            break
        
        print('------------------------------')
        print(f'Processing frame #{frame_counter}')

        #################################
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from frame
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
        if 'load_image' in exec_list:
            front_image = tools.extract_front_camera_image(frame) 

        ## Compute LiDAR point-cloud from range image
        if 'pcl_from_rangeimage' in exec_list:
            print('Computing point-cloud from LiDAR range image')
            lidar_point_cloud = tools.pcl_from_range_image(frame, lidar_name)
        else:
            print('Loading LiDAR point-cloud from result file')
            lidar_point_cloud = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', frame_counter)
            
        ## Compute LiDAR birds-eye view (BEV)
        if 'bev_from_pcl' in exec_list:
            print('Computing birds-eye view from LiDAR point-cloud')
            lidar_bev = pcl.bev_from_pcl(lidar_point_cloud, det_config)
        else:
            print('Loading birds-eye view from result file')
            lidar_bev = load_object_from_file(results_fullpath, data_filename, 'lidar_bev', frame_counter)

        ## 3D object detection
        if det_config.use_labels_as_objects:
            print('Using groundtruth labels as objects')
            detected_objects = tools.convert_labels_into_objects(frame.laser_labels, det_config)
        else:
            if 'detect_objects' in exec_list:
                print('Detecting objects in LiDAR point-cloud')   
                detected_objects = det.detect_objects(lidar_bev, det_model, det_config)
            else:
                print('Loading detected objects from result file')
                detected_objects = load_object_from_file(results_fullpath, data_filename, f'detections_{det_config.arch}_{det_config.conf_thresh}', frame_counter)

        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("Validating object labels")
            valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_point_cloud, det_config, 0 if det_config.use_labels_as_objects else 10)
        else:
            print('Loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(results_fullpath, data_filename, 'valid_labels', frame_counter)            

        ## Performance evaluation for object detection
        if 'measure_detection_performance' in exec_list:
            print('Measuring detection performance')
            detection_performance = eval.measure_detection_performance(detected_objects, frame.laser_labels, valid_label_flags, det_config.min_iou)     
        else:
            print('Loading detection performance measures from file')
            detection_performance = load_object_from_file(results_fullpath, data_filename, f'det_performance_{det_config.arch}_{det_config.conf_thresh}', frame_counter)   

        detection_performance_all.append(detection_performance)  # Store all evaluation results in a list for performance assessment at the end
        
        ## Visualization for object detection
        if 'show_range_image' in exec_list:
            range_image = pcl.show_range_image(frame, lidar_name)
            range_image = range_image.astype(np.uint8)
            cv2.imshow('Range Image', range_image)
            cv2.waitKey(visualization_pause_time)

        if 'show_pcl' in exec_list:
            pcl.show_pcl(lidar_point_cloud)

        if 'show_bev' in exec_list:
            tools.show_bev(lidar_bev, det_config)  
            cv2.waitKey(visualization_pause_time)          

        if 'show_labels_in_image' in exec_list:
            labeled_image = tools.project_labels_into_camera(camera_calibration, front_image, frame.laser_labels, valid_label_flags, 0.5)
            cv2.imshow('Labels in Image', labeled_image)
            cv2.waitKey(visualization_pause_time)

        if 'show_objects_and_labels_in_bev' in exec_list:
            tools.show_objects_labels_in_bev(detected_objects, frame.laser_labels, lidar_bev, det_config)
            cv2.waitKey(visualization_pause_time)         

        if 'show_objects_in_bev_labels_in_camera' in exec_list:
            tools.show_objects_in_bev_labels_in_camera(detected_objects, lidar_bev, front_image, frame.laser_labels, valid_label_flags, camera_calibration, det_config)
            cv2.waitKey(visualization_pause_time)               

        #################################
        ## Perform tracking
        if 'perform_tracking' in exec_list:
            # Set up sensor objects
            if lidar_sensor is None:
                lidar_sensor = Sensor('lidar', lidar_calibration)
            if camera_sensor is None:
                camera_sensor = Sensor('camera', camera_calibration)
            
            # Preprocess LiDAR detections
            lidar_measurements = []
            for detection in detected_objects:
                lidar_measurements = lidar_sensor.generate_measurement(frame_counter, detection[1:], lidar_measurements)
                
            # Preprocess camera detections
            camera_measurements = []
            for label in frame.camera_labels[0].labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    box = label.box
                    # Use camera labels as measurements and add some random noise
                    measurement = [box.center_x, box.center_y, box.width, box.length]
                    measurement[0] += np.random.normal(0, params.sigma_cam_i)
                    measurement[1] += np.random.normal(0, params.sigma_cam_j)
                    camera_measurements = camera_sensor.generate_measurement(frame_counter, measurement, camera_measurements)
            
            # Kalman prediction
            for track in track_manager.track_list:
                print(f'Predicting track {track.id}')
                kalman_filter.predict(track)
                track.set_t((frame_counter - 1) * 0.1)  # Save next timestamp
                
            # Associate all LiDAR measurements to all tracks
            data_association.associate_and_update(track_manager, lidar_measurements, kalman_filter)
            
            # Associate all camera measurements to all tracks
            data_association.associate_and_update(track_manager, camera_measurements, kalman_filter)
            
            # Save results for evaluation
            result_dict = {track.id: track for track in track_manager.track_list}
            track_manager.result_list.append(copy.deepcopy(result_dict))
            label_list = [frame.laser_labels, valid_label_flags]
            all_labels.append(label_list)
            
            # Visualization
            if 'show_tracks' in exec_list:
                fig, ax, ax2 = plot_tracks(fig, ax, ax2, track_manager.track_list, lidar_measurements, frame.laser_labels, valid_label_flags, front_image, camera_sensor, det_config)
                if 'make_tracking_movie' in exec_list:
                    # Save track plots to file
                    filename = f'{results_fullpath}/tracking{frame_counter:03d}.png'
                    print(f'Saving frame {filename}')
                    fig.savefig(filename)

        # Increment frame counter
        frame_counter += 1    

    except StopIteration:
        # If StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break

#################################
## Post-processing

## Evaluate object detection performance
if 'show_detection_performance' in exec_list:
    eval.compute_performance_stats(detection_performance_all, det_config)

## Plot RMSE for all tracks
if 'show_tracks' in exec_list:
    plot_rmse(track_manager, all_labels)

## Make movie from tracking results    
if 'make_tracking_movie' in exec_list:
    make_movie(results_fullpath)
