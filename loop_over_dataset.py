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
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
show_only_frames = [0, 200]  # Show only frames in interval for debugging

## Prepare Waymo Open Dataset file for loading
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename)
model = "darknet"
sequence = "3"
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # Initialize dataset iterator

## Initialize object detection
configs_det = det.load_configs(model_name='darknet')  # Options are 'darknet', 'fpn_resnet'
model_det = det.create_model(configs_det)

configs_det.use_labels_as_objects = False  # True = use groundtruth labels as objects, False = use model-based detection

## Uncomment this setting to restrict the y-range in the final project
# configs_det.lim_y = [-25, 25] 

## Initialize tracking
KF = Filter()  # Set up Kalman filter 
association = Association()  # Init data association
manager = Trackmanagement()  # Init track manager
lidar = None  # Init lidar sensor object
camera = None  # Init camera sensor object
np.random.seed(10)  # Make random values predictable

## Selective execution and visualization
exec_detection = ['pcl_from_rangeimage', 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']  # Options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking = ['perform_tracking']  # Options are 'perform_tracking'
exec_visualization = ['show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie']  # Options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0  # Set pause time between frames in ms (0 = stop between frames until key is pressed)

##################
## Perform detection & tracking over all selected frames

cnt_frame = 0
all_labels = []
det_performance_all = []
if 'show_tracks' in exec_list:
    fig, (ax2, ax) = plt.subplots(1, 2)  # Init track plot

while True:
    try:
        ## Get next frame from Waymo dataset
        frame = next(datafile_iter)
        if cnt_frame < show_only_frames[0]:
            cnt_frame += 1
            continue
        elif cnt_frame > show_only_frames[1]:
            print('Reached end of selected frames')
            break
        
        print('------------------------------')
        print('Processing frame #' + str(cnt_frame))

        #################################
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from frame
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
        if 'load_image' in exec_list:
            image = tools.extract_front_camera_image(frame) 

        ## Compute LiDAR point-cloud from range image
        if 'pcl_from_rangeimage' in exec_list:
            print('Computing point-cloud from LiDAR range image')
            lidar_pcl = tools.pcl_from_range_image(frame, lidar_name)
        else:
            print('Loading LiDAR point-cloud from result file')
            lidar_pcl = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', cnt_frame)
            
        ## Compute LiDAR birds-eye view (BEV)
        if 'bev_from_pcl' in exec_list:
            print('Computing birds-eye view from LiDAR point-cloud')
            lidar_bev = pcl.bev_from_pcl(lidar_pcl, configs_det)
        else:
            print('Loading birds-eye view from result file')
            lidar_bev = load_object_from_file(results_fullpath, data_filename, 'lidar_bev', cnt_frame)

        ## 3D object detection
        if configs_det.use_labels_as_objects:
            print('Using groundtruth labels as objects')
            detections = tools.convert_labels_into_objects(frame.laser_labels, configs_det)
        else:
            if 'detect_objects' in exec_list:
                print('Detecting objects in LiDAR point-cloud')   
                detections = det.detect_objects(lidar_bev, model_det, configs_det)
            else:
                print('Loading detected objects from result file')
                detections = load_object_from_file(results_fullpath, data_filename, 'detections_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)

        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("Validating object labels")
            valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects else 10)
        else:
            print('Loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(results_fullpath, data_filename, 'valid_labels', cnt_frame)            

        ## Performance evaluation for object detection
        if 'measure_detection_performance' in exec_list:
            print('Measuring detection performance')
            det_performance = eval.measure_detection_performance(detections, frame.laser_labels, valid_label_flags, configs_det.min_iou)     
        else:
            print('Loading detection performance measures from file')
            det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)   

        det_performance_all.append(det_performance)  # Store all evaluation results in a list for performance assessment at the end
        
        ## Visualization for object detection
        if 'show_range_image' in exec_list:
            img_range = pcl.show_range_image(frame, lidar_name)
            img_range = img_range.astype(np.uint8)
            cv2.imshow('Range Image', img_range)
            cv2.waitKey(vis_pause_time)

        if 'show_pcl' in exec_list:
            pcl.show_pcl(lidar_pcl)

        if 'show_bev' in exec_list:
            tools.show_bev(lidar_bev, configs_det)  
            cv2.waitKey(vis_pause_time)          

        if 'show_labels_in_image' in exec_list:
            img_labels = tools.project_labels_into_camera(camera_calibration, image, frame.laser_labels, valid_label_flags, 0.5)
            cv2.imshow('Labels in Image', img_labels)
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
            # Set up sensor objects
            if lidar is None:
                lidar = Sensor('lidar', lidar_calibration)
            if camera is None:
                camera = Sensor('camera', camera_calibration)
            
            # Preprocess LiDAR detections
            meas_list_lidar = []
            for detection in detections:
                meas_list_lidar = lidar.generate_measurement(cnt_frame, detection[1:], meas_list_lidar)
                
            # Preprocess camera detections
            meas_list_cam = []
            for label in frame.camera_labels[0].labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    box = label.box
                    # Use camera labels as measurements and add some random noise
                    z = [box.center_x, box.center_y, box.width, box.length]
                    z[0] += np.random.normal(0, params.sigma_cam_i)
                    z[1] += np.random.normal(0, params.sigma_cam_j)
                    meas_list_cam = camera.generate_measurement(cnt_frame, z, meas_list_cam)
            
            # Kalman prediction
            for track in manager.track_list:
                print('Predict track', track.id)
                KF.predict(track)
                track.set_t((cnt_frame - 1) * 0.1)  # Save next timestamp
                
            # Associate all LiDAR measurements to all tracks
            association.associate_and_update(manager, meas_list_lidar, KF)
            
            # Associate all camera measurements to all tracks
            association.associate_and_update(manager, meas_list_cam, KF)
            
            # Save results for evaluation
            result_dict = {track.id: track for track in manager.track_list}
            manager.result_list.append(copy.deepcopy(result_dict))
            label_list = [frame.laser_labels, valid_label_flags]
            all_labels.append(label_list)
            
            # Visualization
            if 'show_tracks' in exec_list:
                fig, ax, ax2 = plot_tracks(fig, ax, ax2, manager.track_list, meas_list_lidar, frame.laser_labels, valid_label_flags, image, camera, configs_det)
                if 'make_tracking_movie' in exec_list:
                    # Save track plots to file
                    fname = results_fullpath + '/tracking%03d.png' % cnt_frame
                    print('Saving frame', fname)
                    fig.savefig(fname)

        # Increment frame counter
        cnt_frame += 1    

    except StopIteration:
        # If StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break

#################################
## Post-processing

## Evaluate object detection performance
if 'show_detection_performance' in exec_list:
    eval.compute_performance_stats(det_performance_all, configs_det)

## Plot RMSE for all tracks
if 'show_tracks' in exec_list:
    plot_rmse(manager, all_labels)

## Make movie from tracking results    
if 'make_tracking_movie' in exec_list:
    make_movie(results_fullpath)
