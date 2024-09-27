# Track 3D-Objects Over Time

## Project Overview

This project focuses on tracking 3D objects over time using the Waymo Open Dataset. The aim is to detect and track objects within LiDAR point-cloud data, incorporating data from camera sensors to improve tracking accuracy. The project is divided into several key steps: filtering, track management, association, and camera fusion.



## Project Instructions Step 1
![Image 1](img\Step1\tracking150.png)
![Image 2](img\Step1\tracking151.png)
![Image 3](img\Step1\tracking152.png)
![Image 4](img\Step1\tracking153.png)
![Image 5 run RMS](img\Step1\Final_run.PNG)

## Project Instructions Step 2
![Image 1](img\Step2\tracking065.png)
![Image 2](img\Step2\tracking066.png)

## Project Instructions Step 3
![Image 1](img\Step3\tracking000.png)
![Image 2](img\Step3\tracking001.png)
![Image 3](img\Step3\tracking002.png)
![Image 4](img\Step3\tracking003.png)

## Project Instructions Step 4
![Video out put](img\Step4\my_tracking_results.avi)
![Video out put](img\Step4\Final.PNG)

### Recap of the Four Tracking Steps

1. **Extended Kalman Filter (EKF)**:
   - Implemented EKF to estimate the state of tracked objects over time, combining measurements from both sensors and accounting for non-linear motion. This step allowed for smooth tracking despite noise and measurement uncertainty.

2. **Track Management**:
   - Developed a system to maintain and update the state of multiple tracks. This included initializing new tracks, updating existing ones based on new observations, and removing tracks that are no longer relevant. This ensured efficient use of resources and accurate tracking of moving objects.

3. **Data Association**:
   - Implemented a nearest neighbor approach to associate detected objects with existing tracks. This step involved matching observations from sensors to the predicted states of the tracks, ensuring that the correct measurements were linked to the right objects.

4. **Camera-Lidar Sensor Fusion**:
   - Combined data from both camera and lidar sensors to improve tracking accuracy. This involved aligning the data, merging measurements, and leveraging the strengths of each sensor to create a more robust tracking system. The result was a significant enhancement in object detection and classification.

### Achievements
The implementation resulted in improved tracking accuracy, with a notable reduction in false positives and false negatives compared to lidar-only tracking. The fusion of data allowed for better handling of occlusions and dynamic environments.

### Most Difficult Part
The most challenging aspect of the project was the **data association** step. Developing an efficient algorithm that reliably matched tracks with observations in real-time was complex, particularly in scenarios with multiple overlapping objects and varying motion patterns.

### Benefits of Camera-Lidar Fusion
The benefits of camera-lidar fusion compared to lidar-only tracking include:
- **Enhanced Object Recognition**: Cameras provide color and texture information, which can improve classification accuracy.
- **Better Handling of Occlusions**: Lidar can struggle in occluded environments, whereas camera data can help fill in gaps.
- **Rich Contextual Information**: Combining the two sources can lead to more comprehensive situational awareness.

### Challenges in Real-Life Scenarios
In real-life scenarios, a sensor fusion system may face challenges such as:
- **Calibration Issues**: Ensuring that the sensors are accurately aligned can be difficult.
- **Environmental Variability**: Changes in lighting or weather conditions can affect sensor performance.
- **Computational Complexity**: Real-time processing of fused data can be resource-intensive.

During the project, I encountered calibration issues that affected the accuracy of the sensor fusion, particularly when aligning lidar and camera data.

### Future Improvements
To improve tracking results in the future, I could explore:
- **Advanced Data Association Techniques**: Implementing algorithms like Hungarian or Munkres for more accurate matching.
- **Deep Learning Approaches**: Utilizing neural networks for feature extraction and data association could enhance performance.
- **Adaptive Sensor Fusion**: Developing methods that dynamically adjust the weighting of each sensor's contribution based on environmental conditions.

These enhancements could lead to even more robust and accurate tracking systems in varied real-world conditions.



## reference
- **Github**
https://github.com/udacity/nd013-c2-fusion-starter.git
https://github.com/adamdivak/udacity_sd_lidar_fusion.git
https://github.com/mabhi16/3D_Object_detection_midterm.git
https://github.com/polarbeargo/nd013-Mid-Term-Project-3D-Object-Detection.git

- **ChatGPT**
- **Tabnine**
- **Mentor Udacity**
- **Mentor StackOverflow**