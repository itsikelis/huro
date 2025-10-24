FROM ros:humble

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="UTC"
ENV ROS_DISTRO=humble
ENV DISPLAY=:0
ENV LIBGL_ALWAYS_INDIRECT=0

WORKDIR /deps

RUN apt-get update && apt-get install -yy \
    python3-pip \
    python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rosidl-generator-dds-idl \
    libpcl-dev \
    terminator \
    nano \
    neovim \
    git \
    gedit \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libwayland-dev \
    libxkbcommon-dev \
    iproute2 \
    libhidapi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pyspacemouse

# MuJoCo
RUN git clone https://github.com/deepmind/mujoco.git && \
    cd mujoco && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . && \
    cmake --install .

# Livox lidar
# RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
#     cd Livox-SDK2 && \
#     mkdir build && \
#     cd build && \
#     cmake .. && \
#     make -j && \
#     make install

WORKDIR /huro_ws/src
RUN git clone https://github.com/ros2/rmw_cyclonedds -b ${ROS_DISTRO} && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd ../ && \
    colcon build --packages-up-to cyclonedds

RUN git clone https://github.com/unitreerobotics/unitree_ros2 && \
    mv unitree_ros2/cyclonedds_ws/src/unitree /huro_ws/src && \
    rm -rf unitree_ros2

# RUN git clone https://github.com/Livox-SDK/livox_ros_driver2.git && \
#     cd livox_ros_driver2 && \
#     rm -f package.xml && \
#     cp package_ROS2.xml package.xml && \
#     cp -rf launch_ROS2 launch

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && echo "source /huro_ws/install/setup.bash" >> /root/.bashrc

RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc

CMD ["bash"]
