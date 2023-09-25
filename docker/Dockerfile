ARG ROS_DISTRO
ARG BASE_IMAGE
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as cuda
FROM $BASE_IMAGE

RUN /bin/bash -c "apt-get update && apt-get -y dist-upgrade"

# Add cuda&cudnn libraries
COPY --from=cuda /usr/local/cuda /usr/local/cuda
COPY --from=cuda /usr/lib/x86_64-linux-gnu/*libcudnn* /usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/targets/x86_64-linux/lib/"
ENV PATH="${PATH}:/usr/local/cuda/bin"

# Install dependencies
COPY . /opt/visual_robot_localization/src
WORKDIR /opt/visual_robot_localization/src
RUN /bin/bash -c "chmod +x install_dependencies.sh; \
		  ./install_dependencies.sh"

# Install hloc
WORKDIR /opt/visual_robot_localization/src/third_party/hloc
RUN /bin/bash -c "chmod +x install_additional_dependencies.sh; \
		   bash install_additional_dependencies.sh"	
RUN /bin/bash -c "python3 setup.py install"

# Build the package
WORKDIR /opt/visual_robot_localization/
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; \
		  colcon build" 
		  
COPY ./docker/content/ros_entrypoint.sh /
