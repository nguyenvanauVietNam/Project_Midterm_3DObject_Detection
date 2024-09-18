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
