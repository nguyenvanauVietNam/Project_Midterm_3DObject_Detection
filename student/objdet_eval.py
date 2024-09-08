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
#

# General package imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon
from operator import itemgetter

# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Object detection tools and helper functions
import misc.objdet_tools as tools


# Compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
    # Find best detection for each valid label 
    true_positives = 0  # Number of correctly detected objects
    center_devs = []
    ious = []
    
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:  # Exclude all labels from statistics which are not considered valid
            
            # Compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######
            #######
            print("student task ID_S4_EX1")

            # Step 1: Extract the four corners of the current label bounding-box
            label_polygon = Polygon([
                (label[1], label[2]),
                (label[1] + label[4], label[2]),
                (label[1] + label[4], label[2] + label[5]),
                (label[1], label[2] + label[5])
            ])

            # Step 2: Loop over all detected objects
            for detection in detections:
                # Extract the four corners of the detection bounding-box
                det_polygon = Polygon([
                    (detection[1], detection[2]),
                    (detection[1] + detection[4], detection[2]),
                    (detection[1] + detection[4], detection[2] + detection[5]),
                    (detection[1], detection[2] + detection[5])
                ])

                # Step 3: Compute the center distance between label and detection bounding-box in x, y, and z
                center_dist_x = abs((label[1] + label[4]/2) - (detection[1] + detection[4]/2))
                center_dist_y = abs((label[2] + label[5]/2) - (detection[2] + detection[5]/2))
                center_dist_z = 0.0  # Assuming 2D for simplicity

                # Step 4: Compute the intersection over union (IOU) between label and detection bounding-box
                iou = label_polygon.intersection(det_polygon).area / label_polygon.union(det_polygon).area

                # Step 5: If IOU exceeds min_iou threshold, store [iou, dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if iou > min_iou:
                    matches_lab_det.append([iou, center_dist_x, center_dist_y, center_dist_z])
                    true_positives += 1
                
            ####### ID_S4_EX1 END #######     
            
        # Find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det, key=itemgetter(0))  # Retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # Compute positives and negatives for precision/recall
    
    # Step 1: Compute the total number of positives present in the scene
    all_positives = sum(labels_valid)
    
    # Step 2: Compute the number of false negatives
    false_negatives = all_positives - true_positives

    # Step 3: Compute the number of false positives
    false_positives = len(detections) - true_positives
    
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# Evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # Extract elements
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

    # Step 1: Extract the total number of positives, true positives, false negatives and false positives
    all_positives = sum([item[0] for item in pos_negs])
    true_positives = sum([item[1] for item in pos_negs])
    false_negatives = sum([item[2] for item in pos_negs])
    false_positives = sum([item[3] for item in pos_negs])
    
    # Step 2: Compute precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    # Step 3: Compute recall 
    recall = true_positives / all_positives if all_positives > 0 else 0.0

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # Serialize intersection-over-union and deviations in x, y, z
    ious_all = [element for sublist in ious for element in sublist]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    
    for sublist in center_devs:
        for elem in sublist:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)

    # Compute statistics
    stdev_ious = np.std(ious_all)
    mean_ious = np.mean(ious_all)

    stdev_devx = np.std(devs_x_all)
    mean_devx = np.mean(devs_x_all)

    stdev_devy = np.std(devs_y_all)
    mean_devy = np.mean(devs_y_all)

    stdev_devz = np.std(devs_z_all)
    mean_devz = np.mean(devs_z_all)

    # Plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['Detection precision', 'Detection recall', 'Intersection over union', 'Position errors in X', 'Position errors in Y', 'Position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (mean_devx, ), r'$\mathrm{sigma}=%.4f$' % (stdev_devx, ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (mean_devy, ), r'$\mathrm{sigma}=%.4f$' % (stdev_devy, ), r'$\mathrm{n}=%.0f$' % (len(devs_y_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (mean_devz, ), r'$\mathrm{sigma}=%.4f$' % (stdev_devz, ), r'$\mathrm{n}=%.0f$' % (len(devs_z_all), )))]

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
