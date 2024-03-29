# Visual robot localization

A ROS2 package for hierarchical visual localization.

<p align="center">
  <img src="doc/vloc_scheme.png" width="60%"/></a>
  <br /><em>An agent running ROS2 can acquire its 6DoF pose using hierarchical visual localization</em>
</p>

## Citing

If you find this repository useful in your research, please cite our work as:
```
@InProceedings{Suomela_2023_WACV,
    author    = {Suomela, Lauri and Kalliola, Jussi and Dag, Atakan and Edelman, Harry and Kämäräinen, Joni-Kristian},
    title     = {Benchmarking Visual Localization for Autonomous Navigation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {2945-2955}
}
```

## Install

Pull the repository:

```sh
git clone https://github.com/lasuomela/visual_robot_localization/
git submodule update --init --recursive
```

The easiest way to test the package is to run the ROS2 node inside the provided docker containter:

```sh
cd docker
./build.sh
./run.sh
cd /opt/visual_robot_localization
colcon build
```
Alternatively, you can install the package and system dependencies into your local environment by following the instuctions in the `Dockerfile` and `install_dependencies.sh`.

## Launch testing
Now, navigate to `/opt/visual_robot_localization/src/visual_robot_localization/utils` and run

```sh
./do_SfM.sh
```

This will run a 3D reconstruction from the images and camera poses located in `visual_robot_localization/test/example_dir`. You can alter the parameters of the script to choose different visual localization methods.

If the reconstruction is successful, you can now run 

```sh
./visualize_colmap.sh
```

to visualize the scene reconstruction in COLMAP.

After running the 3D reconstruction, you can launch the visual localizer with

```sh
ros2 launch visual_robot_localization visual_pose_estimator.launch.py compensate_sensor_offset:=False
```

If `compensate_sensor_offset` is set to True, the node will wait to acquire a coordinate transform between the camera and robot base from tf2. Since tf2 isn't running, the parameter is set to False.

To test that the node can succesfully receive and process images, run 

```sh
launch_test /opt/visual_robot_localization/src/visual_robot_localization/test/visual_pose_estimator_test.launch.py
```

The test sends the visual pose estimator node one image from `example_dir` which the node localizes against the 3D model built by `do_SfM.sh`. If the localization is successful, the test will give an OK and show one visual pose estimate ROS message 

## Workflow

Usage of the visual localization package follows the steps described below:

1. Collect a gallery set describing the environment
2. Run scene structure reconstruction with the desired visual localization methods using `./do_SfM.sh`
3. Launch the visual localizer node using `ros2 launch visual_robot_localization visual_pose_estimator.launch.py`

### Data format

The package expects the gallery images in `.jpg` or `.png` format. For each image, the pose of the camera should be stored in a `.json` which follows the structure of ROS nav_msgs/Odometry. If the image is named `im_0001.png`, the `.json` file should be named `im_0001_odometry_camera.json`. See `visual_robot_localization/test/example_dir` for an example.

### Visual pose estimator node

Interface description

### Place recognition node

Interface description

## Extending

Under the hood, the package utilizes the [hloc toolbox](https://github.com/cvg/Hierarchical-Localization) for integrating visual localization methods, batch feature extraction and scene reconstruction. New visual localization methods can be added by contributing to the hloc package.

## TODO

- [ ] Enable running scene reconstruction with full SfM with unkown camera poses
- [ ] Improve test coverage
