# visual_robot_localization

A ROS2 package for running hierarchical visual localization on robotic systems.

## Getting started

Clone the repository, and in te repository run


```sh
git submodule update --init --recursive
```

The easiest way to test the package is to run the ROS2 node inside the provided docker containter.

To build, run the

```sh
./build.sh
```

script in the docker directory. After a successful build,

```sh
./run.sh
```
starts the container.

To build the ros package, run
```sh
colcon build
```

inside the `/opt/visual_robot_localization` directory.

Now, navigate to `/opt/visual_robot_localization/src/visual_robot_localization/utils` and run

```sh
./do_SfM.sh
```

This will run a 3D reconstruction from the images and camera poses located in `visual_robot_localization/test/example_dir`. You can alter the parameters of the script to choose different visual localization methods.

If the reconstruction is successful, you can now run 

```sh
./visualize_colmap.sh
```

to visualize the scene reconstruction.

## Visual localization node

After running the 3D reconstruction, you can launch the visual localizer with

```sh
ros2 launch visual_robot_localization visual_pose_estimator.launch.py compensate_sensor_offset:=False
```

If `compensate_sensor_offset` is set to True, the node will wait to acquire a coordinate transform between the camera and robot base from tf2. Since tf2 isn't running, the parameter is set to False.

To test that the node can succesfully receive and process images, navigate to `/opt/visual_robot_localization/src/visual_robot_localization/test` and run 

```sh
launch_testing visual_pose_estimator_test.launch.py
```

The test sends the visual pose estimator node one image from `example_dir` which the node localizes against the 3D model built by `do_SfM.sh`. If the localization is successful, the test will give an OK and show one visual pose estimate ROS message 
