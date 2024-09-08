# Track 3D-Objects Over Time

## Project Overview

This project focuses on tracking 3D objects over time using the Waymo Open Dataset. The aim is to detect and track objects within LiDAR point-cloud data, incorporating data from camera sensors to improve tracking accuracy. The project is divided into several key steps: filtering, track management, association, and camera fusion.


## Step-1: Compute Lidar point cloud from Range Image
- **Visualize range image channels (ID_S1_EX1)**
In file loop_over_dataset.py, set the attributes for code execution in the following way:
![loop_over_dataset.py](img\Project.png)
- **What your result should look like**
![img](img\range-image-viz.png)

## 3. Challenges and Difficulties

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
