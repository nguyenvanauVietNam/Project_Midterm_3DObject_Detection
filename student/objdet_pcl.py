# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
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
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d    # import o3d by command auditor
#import dataset_pb2  # import dataset_pb2 from  protobuf

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(point_cloud_data):
    """
    Visualizes the 3D point cloud data using Open3D.

    Parameters:
    - point_cloud_data: Array of 3D points to be visualized.
    """

    ####### ID_S1_EX2 START #######
    #######
    print("Starting visualization task ID_S1_EX2")

    # Step 1: Initialize Open3D visualizer with key callback and create a window
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name='Point Cloud Visualization') 
    
    # Global variable to control window display
    global keep_running
    keep_running = True

    # Define a callback function for the right arrow key (key-code 262) to stop visualization
    def on_right_arrow_key_pressed(vis):
        global keep_running
        print('Right arrow key pressed, closing visualization')
        keep_running = False
        return

    visualizer.register_key_callback(262, on_right_arrow_key_pressed)
    
    # Step 2: Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    
    # Step 3: Set points in PointCloud object by converting the point cloud data into 3D vectors
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
    
    # Step 4: Add the PointCloud object to the visualizer
    visualizer.add_geometry(point_cloud)
    
    # Step 5: Start visualization and keep the window open until the right-arrow key is pressed
    while keep_running:
        visualizer.poll_events()
        visualizer.update_renderer()

    # Clean up and close the visualizer window
    visualizer.destroy_window()
    #######
    ####### ID_S1_EX2 END #######

    
       


# visualize range image
def show_range_image(frame, lidar_name):
    """
    Extracts and visualizes the range and intensity channels from a specified lidar sensor's range image.

    Parameters:
    - frame: The frame containing lidar data.
    - lidar_name: The name of the lidar sensor from which to extract data.

    Returns:
    - img_combined: The stacked and cropped image of range and intensity channels.
    """

    print("Processing range image for lidar sensor:", lidar_name)

    # Step 1: Extract the lidar data object for the specified lidar_name from the frame
    lidar_data = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    
    # Step 2: Extract and decompress the range image data from the lidar data object
    if len(lidar_data.ri_return1.range_image_compressed) > 0:  # Use the first response
        # Decompress the range image data
        decompressed_image_data = zlib.decompress(lidar_data.ri_return1.range_image_compressed)
        
        # Parse the decompressed data into a MatrixFloat object and convert it to a numpy array
        range_image_matrix = dataset_pb2.MatrixFloat()
        range_image_matrix.ParseFromString(decompressed_image_data)
        range_image_data = np.array(range_image_matrix.data).reshape(range_image_matrix.shape.dims)
    else:
        print("Can't receive range image from", lidar_name, "lidar sensor")
    
    # Step 3: Set values less than 0 to zero in the range image data
    range_image_data[range_image_data < 0] = 0.0

    # Step 4: Map the range channel to an 8-bit scale
    range_channel_data = range_image_data[:, :, 0]
    range_channel_data = range_channel_data * 255 / (np.amax(range_channel_data) - np.amin(range_channel_data))
    img_range = range_channel_data.astype(np.uint8)

    # Step 5: Map the intensity channel to an 8-bit scale and normalize
    intensity_channel_data = range_image_data[:, :, 1]
    intensity_channel_data = np.amax(intensity_channel_data) / 2 * intensity_channel_data * 255 / (np.amax(intensity_channel_data) - np.amin(intensity_channel_data))
    img_intensity = intensity_channel_data.astype(np.uint8)

    # Step 6: Stack the range and intensity images vertically
    img_combined = np.vstack((img_range, img_intensity))

    # Step 7: Crop the central region of the stacked image
    center_x = img_combined.shape[1] // 2
    crop_width = img_combined.shape[1] // 4
    img_combined = img_combined[:, center_x - crop_width:center_x + crop_width]

    return img_combined


# create birds-eye view of lidar data
def bev_from_pcl(lidar_point_cloud, configs):


    # remove lidar points outside detection area and with too low reflectivity
    mask  = np.where(
        (lidar_point_cloud[:, 0] >= configs.lim_x[0]) & (lidar_point_cloud[:, 0] <= configs.lim_x[1]) &
        (lidar_point_cloud[:, 1] >= configs.lim_y[0]) & (lidar_point_cloud[:, 1] <= configs.lim_y[1]) &
        (lidar_point_cloud[:, 2] >= configs.lim_z[0]) & (lidar_point_cloud[:, 2] <= configs.lim_z[1])
    )
    lidar_pcl  = lidar_point_cloud[mask]

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl [:, 2] = lidar_pcl [:, 2] - configs.lim_z[0]
    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")
    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    print("Processing BEV map from lidar data...")
    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates 
    lidar_pcl_bev = np.copy(lidar_pcl )
    lidar_pcl_bev[:, 0] = np.int_(np.floor(lidar_pcl_bev[:, 0] / bev_discretization))
    lidar_pcl_bev[:, 1] = np.int_(np.floor(lidar_pcl_bev[:, 1] / bev_discretization)) + (configs.bev_width + 1) / 2
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    #   lidar_pcl_bev[:, 2] = lidar_pcl_bev[:, 2] / bev_discretization

    # Visualize the transformed point cloud (for debugging purposes)
    show_pcl(lidar_pcl_bev)
    #######
    ####### ID_S2_EX1 END #######  

    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    # Compute intensity layer of the BEV map
    print("Computing intensity map...")
    print("student task ID_S2_EX2")
    #reference https://github.com/polarbeargo/nd013-Mid-Term-Project-3D-Object-Detection/blob/main/student/objdet_pcl.py#L76
    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height +3 , configs.bev_width +3 ))
    
   # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    lidar_pcl_bev [:, 3] = np.clip(lidar_pcl_bev [:, 3], 0, 1.0)
    int_upper = np.quantile(lidar_pcl_bev[:,3], 0.95)
    lidar_pcl_bev[lidar_pcl_bev[:, 3] > int_upper, 3] = int_upper

    if(lidar_pcl_bev[lidar_pcl_bev[:,3]>1.0,3] != 1.0):
        lidar_pcl_bev[lidar_pcl_bev[:,3]>1.0,3] = 1.0

    idx = np.lexsort((-lidar_pcl_bev[:, 3], lidar_pcl_bev[:, 1], lidar_pcl_bev[:, 0]))
    lidar_pcl_top = lidar_pcl_bev[idx]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _,lidar_pcl_int, indxx, count = np.unique(lidar_pcl_bev[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_bev[indxx]

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    intensity_map = np.zeros((configs.bev_height + 3, configs.bev_width + 3))
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3])-np.amin(lidar_pcl_top[:, 3]))
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    image_intensity = intensity_map * 256
    image_intensity = image_intensity.astype(np.uint8)
    cv2.imshow('Images_Intensity', image_intensity)
    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height + 3, configs.bev_width + 3))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    
    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background

    #images_height = height_map * 256
    images_height = images_height.astype(np.uint8)
    cv2.imshow('images_height', images_height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX3 END #######  


    # TODO remove after implementing all of the above steps
    # lidar_pcl_cpy = []
    # lidar_pcl_top = []
    # height_map = []
    # intensity_map = []
     
    
    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_bev[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps

