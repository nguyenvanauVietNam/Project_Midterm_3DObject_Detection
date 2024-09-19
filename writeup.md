# Track 3D-Objects Over Time

## Project Overview

This project focuses on tracking 3D objects over time using the Waymo Open Dataset. The aim is to detect and track objects within LiDAR point-cloud data, incorporating data from camera sensors to improve tracking accuracy. The project is divided into several key steps: filtering, track management, association, and camera fusion.


## Step-1: Compute Lidar point cloud from Range Image
- **Visualize range image channels (ID_S1_EX1)**
In file loop_over_dataset.py, set the attributes for code execution in the following way:
![loop_over_dataset.py](img\Project.png)
- **What your result should look like**
![img](img\range-image-viz.png)

--------
## Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
show_only_frames = [0, 1] # show only frames in interval for debugging
## Selective execution and visualization
exec_data = []
exec_detection = []
exec_tracking = [] 
exec_visualization = ['show_range_image'] 
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 
- **My result**
![img](img\Range_image_ID_S1_EX1_AuNV.png)
--------
- **Visualize range image channels (ID_S1_EX2)**
- **What your result should look like**
![img](img\point cloud visualization.png)
--------
## Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 2
show_only_frames = [0, 200] # show only frames in interval for debugging
## Selective execution and visualization
exec_data = []
exec_detection = []
exec_tracking = [] 
exec_visualization = ['show_pcl'] 
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 
- **My result**
![img](img\Point_cloud_image_ID_S1_EX2_AuNV.png)
--------

##  Section 2 : Create Birds-Eye View from Lidar PCL
- **Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)**
data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
sequence_num = "2"
show_only_frames = [0, 1] 
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl']
exec_tracking = [] 
exec_visualization = [] 
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 
- **What your result should look like**
![img](img\BEV map coordinates.png)
- **My result**
![img](img\BEV-map coordinates_ID_S2_EX1_AuNV.png)

- **Compute intensity layer of the BEV map (ID_S2_EX2)**
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 2
sequence_num = "2"
show_only_frames = [0, 1] 
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl']
exec_tracking = [] 
exec_visualization = [] 
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 
- **My result**
![img](img\ID_S2_EX2_1.png)
![img](img\ID_S2_EX2_2.png)


- **Compute height layer of the BEV map (ID_S2_EX3)**
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 2
sequence_num = "1"
show_only_frames = [0, 1] 
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl']
exec_tracking = [] 
exec_visualization = [] 
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 


## Section 3 : Model-based Object Detection in BEV Image
- **Add a second model from a GitHub repo (ID_S3_EX1)**
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 2
show_only_frames = [50, 51] 
model_name = "fpn_resnet"
configs_det = det.load_configs(model_name='fpn_resnet') 
exec_data = ['pcl_from_rangeimage', 'load_image']
exec_detection = ['bev_from_pcl', 'detect_objects']
exec_tracking = [] 
exec_visualization = ['show_objects_in_bev_labels_in_camera'] 

## Section 4 : Performance Evaluation for Object Detection
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 2
show_only_frames = [50, 51] 
model_name = "fpn_resnet"
configs_det = det.load_configs(model_name='fpn_resnet') 
exec_data = ['pcl_from_rangeimage', 'load_image']
exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
exec_tracking = [] 
exec_visualization = ['show_detection_performance']
configs_det = det.load_configs(model_name="darknet")
## Section Final
data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
show_only_frames = [170, 200] # show only frames in interval for debugging

### Most Difficult Part
The integration of camera data with LiDAR data was the most challenging due to differences in data formats and calibration requirements. Aligning these sensors accurately was crucial for effective fusion.

### Real-Life Sensor Fusion Challenges
- **Calibration:** Accurate sensor calibration is difficult but essential for effective data fusion.
- **Data Synchronization:** Aligning and synchronizing data from different sensors is complex.
- **Processing Complexity:** The computational load increases with multiple sensors, demanding efficient processing.

## 4. Future Improvements

- **Enhanced Calibration:** Develop advanced calibration techniques for better sensor alignment.
- **Advanced Fusion Algorithms:** Explore sophisticated algorithms to handle inconsistencies between sensor data.
- **Real-Time Processing:** Optimize the system for real-time data processing to improve efficiency.
- **Additional Sensors:** Consider incorporating other sensors, such as radar, for improved tracking accuracy.




## Conclusion

The project successfully demonstrated the capability of tracking 3D objects over time using a combination of LiDAR and camera data. Despite challenges, the integration of multiple sensors enhanced tracking accuracy and robustness, providing valuable insights into real-time object tracking.

# 3D Object Tracking Project Overview

## 1. Recap of the Four Tracking Steps

### Filtering
In the filtering step, we implemented an Extended Kalman Filter (EKF) to track objects. The EKF used:
- **Linear Measurement Model:** For lidar detections.
- **Non-Linear Measurement Model:** For camera detections.

The filter transformed sensor data into the vehicle coordinate system and used a constant velocity model for predictions. The filter's role was to predict future object states (position and velocity) based on past measurements.

**Results:** The EKF effectively managed both lidar and camera data, improving tracking accuracy and stabilizing object trajectories. Despite this, occasional false detections introduced errors in the predictions.

### Track Management
Track management involved:
- Initializing new tracks for unassigned detections.
- Calculating track scores based on successful detections over past frames.
- Managing three track states: init, tentative, and confirmed.

Tracks were deleted if their score fell below a threshold or their uncertainty (covariance) became too large.

**Results:** The track management system successfully maintained object identities and removed unreliable tracks. However, occasional false detections led to unrealistic track behaviors, such as objects moving when they should be stationary.

### Association
Measurement association was performed using Mahalanobis distance, accounting for the means and covariances of tracks and detections. This approach managed noisy and uncertain data effectively.

**Results:** The association method generally linked tracks to measurements accurately, even with erratic object movements. Nevertheless, in crowded scenes, the system struggled, resulting in track switches or mismatches.

### Camera Fusion
Camera fusion involved transforming Kalman Filter’s state estimates from lidar into camera and image space using a non-linear transformation. This combined precise depth information from lidar with semantic information from the camera.

**Results:** Fusion enhanced tracking reliability and accuracy, especially in scenarios where lidar alone was insufficient. Although successful, there were occasional false detections and misclassifications that need further refinement.

**Most Difficult Part:** The most challenging aspect was managing associations and tracks, particularly in crowded scenes. Handling ambiguities and fine-tuning parameters to address these issues proved to be complex.

## 2. Benefits of Camera-Lidar Fusion vs. Lidar-Only Tracking

### Theory
Camera-lidar fusion combines the strengths of both sensors:
- **Lidar:** Provides accurate 3D information on object distance and shape.
- **Camera:** Offers rich color and texture details for better object classification and feature detection.

Lidar alone may not recognize visual cues like brake lights or turn signals that a camera can detect.

### Results
In practice, combining lidar and camera data led to:
- Improved object detection and tracking, especially for partially visible objects.
- Enhanced ability to filter out false detections due to additional scene information.

**Concrete Results:** Fusion resulted in fewer false positives and better object classification compared to using lidar alone.

## 3. Challenges of Sensor Fusion in Real-Life Scenarios

### Sensor Discrepancies
Different sensors may detect objects differently. For instance, a lidar might detect distant objects that a camera misses due to resolution or lighting conditions. Deciding which sensor's data to trust can be challenging.

### Error Propagation
Mistakes in the tracking process can propagate through future frames, complicating recovery. This issue is more pronounced in single-hypothesis systems that follow only one interpretation of the data.

### Synchronization
Perfect synchronization between camera and lidar data is difficult due to varying frame rates. Even minor time misalignments can introduce errors when combining data.

### Environmental Conditions
Real-world environments pose challenges such as weather conditions, changing lighting, and occlusions. These factors can lead to sensor failures or inaccurate detections.

**Challenges Seen in Project:** The project encountered issues with false detections and synchronization between sensors. Small association errors also impacted tracking performance in subsequent frames.

## 4. Ways to Improve Tracking Results in the Future

### Improved Models
Adopting more sophisticated models, like the bicycle model, could improve accuracy by accounting for vehicle dynamics and enforcing realistic motion constraints.

### Better Data Association
Utilizing advanced methods such as Joint Probabilistic Data Association (JPDA) or Multiple Hypothesis Tracking (MHT) could enhance performance in crowded scenes where nearest-neighbor association struggles.

### 360-Degree Sensor Fusion
Incorporating additional sensors (cameras, lidar, radar) for a 360-degree view would improve tracking accuracy by ensuring detection of objects outside any single sensor’s field of view.

### Adding Radar
Integrating radar could improve velocity measurements. Radar provides direct velocity data, enhancing tracking reliability.

### More Robust Track Management
Implementing adaptive thresholding based on scene context could prevent unnecessary track deletions and reinforce valid tracks more effectively.
