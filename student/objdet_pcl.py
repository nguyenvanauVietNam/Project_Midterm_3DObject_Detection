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
    """
    Converts lidar point cloud data into a birds-eye view (BEV) map with intensity, height, and density layers.

    Parameters:
    - lidar_point_cloud: Array of lidar points with x, y, z coordinates and intensity values.
    - configs: Configuration object containing parameters for BEV map generation.

    Returns:
    - input_bev_maps: A tensor representing the BEV map with intensity, height, and density layers.
    """
    
    # Step 1: Filter lidar points based on detection area and reflectivity
    valid_mask = np.where(
        (lidar_point_cloud[:, 0] >= configs.lim_x[0]) & (lidar_point_cloud[:, 0] <= configs.lim_x[1]) &
        (lidar_point_cloud[:, 1] >= configs.lim_y[0]) & (lidar_point_cloud[:, 1] <= configs.lim_y[1]) &
        (lidar_point_cloud[:, 2] >= configs.lim_z[0]) & (lidar_point_cloud[:, 2] <= configs.lim_z[1])
    )
    filtered_lidar = lidar_point_cloud[valid_mask]
    
    # Step 2: Adjust the ground plane level for consistency
    filtered_lidar[:, 2] = filtered_lidar[:, 2] - configs.lim_z[0]
    
    # Step 3: Convert lidar coordinates to BEV map coordinates
    print("Processing BEV map from lidar data...")

    # Compute BEV discretization factor
    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    
    # Transform lidar coordinates to BEV map coordinates
    lidar_pcl_bev = np.copy(filtered_lidar)
    lidar_pcl_bev[:, 0] = np.int_(np.floor(lidar_pcl_bev[:, 0] / bev_discretization))
    lidar_pcl_bev[:, 1] = np.int_(np.floor(lidar_pcl_bev[:, 1] / bev_discretization)) + (configs.bev_width + 1) // 2
    lidar_pcl_bev[:, 1] = np.abs(lidar_pcl_bev[:, 1])
    
    # Visualize the transformed point cloud (for debugging purposes)
    show_pcl(lidar_pcl_bev)

    # Compute intensity layer of the BEV map
    print("Computing intensity map...")
    
    # Initialize intensity map with zeros
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))
    
    # Sort lidar points by x, y, and -z to get the top-most points
    filtered_lidar[:, 3] = np.clip(filtered_lidar[:, 3], 0, 1.0)
    sorted_indices = np.lexsort((-filtered_lidar[:, 2], filtered_lidar[:, 1], filtered_lidar[:, 0]))
    top_lidar_points = filtered_lidar[sorted_indices]
    
    # Keep only the top-most points for each x, y coordinate
    unique_lidar_points, unique_indices, point_counts = np.unique(top_lidar_points[:, :2], axis=0, return_index=True, return_counts=True)
    top_lidar_points = top_lidar_points[unique_indices]
    
    # Normalize intensity values and assign to the intensity map
    intensity_map[np.int_(top_lidar_points[:, 0]), np.int_(top_lidar_points[:, 1])] = (
        top_lidar_points[:, 3] / (np.amax(top_lidar_points[:, 3]) - np.amin(top_lidar_points[:, 3]))
    )
    
    # Visualize the intensity map
    intensity_image = intensity_map * 256
    intensity_image = intensity_image.astype(np.uint8)
    cv2.imshow('Intensity Map', intensity_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Compute height layer of the BEV map
    print("Computing height map...")
    
    # Initialize height map with zeros
    height_map = np.zeros((configs.bev_height, configs.bev_width))
    
    # Normalize height values and assign to the height map
    height_map[np.int_(top_lidar_points[:, 0]), np.int_(top_lidar_points[:, 1])] = (
        top_lidar_points[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    )
    
    # Visualize the height map
    height_image = height_map * 256
    height_image = height_image.astype(np.uint8)
    cv2.imshow('Height Map', height_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Compute density layer of the BEV map
    print("Computing density map...")
    
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, point_counts = np.unique(filtered_lidar[:, :2], axis=0, return_index=True, return_counts=True)
    normalized_density = np.minimum(1.0, np.log(point_counts + 1) / np.log(64))
    density_map[np.int_(top_lidar_points[:, 0]), np.int_(top_lidar_points[:, 1])] = normalized_density
    
    # Assemble the final BEV map with three channels (intensity, height, density)
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # Density map (R channel)
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # Height map (G channel)
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # Intensity map (B channel)
    
    # Convert BEV map to tensor
    bev_map_tensor = np.expand_dims(bev_map, axis=0)
    bev_map_tensor = torch.from_numpy(bev_map_tensor)  # Convert to tensor
    input_bev_maps  = bev_map_tensor.to(configs.device, non_blocking=True).float()
    
    return input_bev_maps
