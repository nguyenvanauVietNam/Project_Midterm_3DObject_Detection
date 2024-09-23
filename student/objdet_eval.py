# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
# https://github.com/udacity/nd013-c2-fusion-starter.git
# https://github.com/adamdivak/udacity_sd_lidar_fusion.git
# https://github.com/mabhi16/3D_Object_detection_midterm.git
# https://github.com/polarbeargo/nd013-Mid-Term-Project-3D-Object-Detection.git
#

# general package imports
import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
    # find best detection for each valid label 
    true_positives = 0  # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:  # exclude all labels from statistics which are not considered valid
            
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            # label_poly = Polygon([(label[1], label[2]), (label[1] + label[4], label[2]),
            #                       (label[1] + label[4], label[2] + label[5]), (label[1], label[2] + label[5])])
            #Fix bug base comment mentor
            label_box = label.box
            label_temp_corners = tools.compute_box_corners(label_box.center_x, label_box.center_y, label_box.width, label_box.length, label_box.heading)
            #label_poly = Polygon(label_temp_corners)

            ## step 2 : loop over all detected objects
            for detection in detections:
                #det_id, x, y, z, h, w, l,yaw = detection
                if len(detection) == 7:
                    det_id,x, y, z, w, l, h ,yaw = detection
                    print(detection)
                    detections_container = tools.compute_box_corners(x, y, w, l, 0)
                    yaw = 0  # Thêm giá trị mặc định cho yaw nếu bị thiếu
                else:
                    det_id, x, y, z, h, w, l, yaw = detection
                    detections_container = tools.compute_box_corners(x, y, w, l, yaw)
                 
                ## step 3 : extract the four corners of the current detection
                # det_poly = Polygon([(temp_x, temp_y), (temp_x + temp_length, temp_y),
                #                     (temp_x + temp_length, temp_y + temp_width), (temp_x, temp_y + temp_width)])

                ## step 4 : compute the center distance between label and detection bounding-box in x, y, and z
                center_dist_x = label_box.center_x - x
                center_dist_y = label_box.center_y - y
                center_dist_z = label_box.center_z - z
                distance_label = np.array([label_box.center_x, label_box.center_y, label_box.center_z]) - np.array([x, y, z])
                
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                # intersection_area = label_poly.intersection(det_poly).area
                # union_area = label_poly.union(det_poly).area

                # iou = intersection_area / union_area if union_area > 0 else 0
                label_poly = Polygon(label_temp_corners)
                detecion_label_poly = Polygon(detections_container)

                inter_label = detecion_label_poly.intersection(label_poly)
                union_label = detecion_label_poly.union(label_poly)
                iou = inter_label.area / union_label.area
                
                ## step 6 : if IOU exceeds min_iou threshold, store [iou, dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if iou > min_iou:
                    center_dist_x, center_dist_y, center_dist_z = distance_label
                    matches_lab_det.append([iou, center_dist_x, center_dist_y, center_dist_z])
                    #true_positives += 1
            
            #######
            ####### ID_S4_EX1 END #######     
            
        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det, key=itemgetter(0))  # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = sum(labels_valid)
    
    ## step 2 : compute the number of false negatives
    false_negatives = all_positives - true_positives

    ## step 3 : compute the number of false positives
    false_positives = len(detections) - true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    all_positives = np.sum([item[0] for item in pos_negs])
    true_positives = np.sum([item[1] for item in pos_negs])
    false_negatives = np.sum([item[2] for item in pos_negs])
    false_positives = np.sum([item[3] for item in pos_negs])
    
    ## step 2 : compute precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    ## step 3 : compute recall 
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev_ious = np.std(ious_all)
    mean_ious = np.mean(ious_all)

    stdev_devx = np.std(devs_x_all)
    mean_devx = np.mean(devs_x_all)

    stdev_devy = np.std(devs_y_all)
    mean_devy = np.mean(devs_y_all)

    stdev_devz = np.std(devs_z_all)
    mean_devz = np.mean(devs_z_all)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (mean_devx, ), r'$\mathrm{sigma}=%.4f$' % (stdev_devx, ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (mean_devy, ), r'$\mathrm{sigma}=%.4f$' % (stdev_devy, ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (mean_devz, ), r'$\mathrm{sigma}=%.4f$' % (stdev_devz, ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]
    
    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()
