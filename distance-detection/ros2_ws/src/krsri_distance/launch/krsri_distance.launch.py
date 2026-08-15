"""
Launch lengkap: Kinect → point cloud → deteksi jarak → RViz.

    ros2 launch krsri_distance krsri_distance.launch.py \
        model_path:=$HOME/krsri/best_ncnn_model

Empat proses yang dinyalakan:
  1. kinect_node           — driver Kinect v1, depth sudah teregistrasi ke RGB
  2. point_cloud_container — depth_image_proc, membuat PointCloud2 berwarna
  3. distance_node         — YOLO + estimasi jarak → Detection3DArray + Marker
  4. rviz2                 — visualisasi (matikan dengan use_rviz:=false)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# Topic yang diterbitkan kinect_node (namespace "camera").
RGB_TOPIC = "/camera/image_raw"
DEPTH_TOPIC = "/camera/depth_registered/image_raw"
INFO_TOPIC = "/camera/camera_info"
POINTS_TOPIC = "/camera/depth_registered/points"


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("krsri_distance")
    default_rviz = os.path.join(package_share, "rviz", "krsri.rviz")

    args = [
        DeclareLaunchArgument(
            "model_path",
            description="Path ke folder best_ncnn_model (atau file .pt).",
        ),
        DeclareLaunchArgument("imgsz", default_value="640"),
        DeclareLaunchArgument("conf", default_value="0.50"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
        DeclareLaunchArgument(
            "frame_id",
            default_value="camera_rgb_optical_frame",
            description="Frame optik: X kanan, Y bawah, Z maju.",
        ),
    ]

    kinect_node = Node(
        package="krsri_distance",
        executable="kinect_node",
        name="kinect_node",
        namespace="camera",
        output="screen",
        parameters=[{"frame_id": LaunchConfiguration("frame_id")}],
    )

    # depth_image_proc menyusun PointCloud2 berwarna dari depth teregistrasi +
    # RGB + intrinsik. Inilah awan titik yang muncul di RViz — bukan kita yang
    # menghitungnya sendiri.
    point_cloud_container = ComposableNodeContainer(
        name="point_cloud_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::PointCloudXyzrgbNode",
                name="point_cloud_xyzrgb",
                remappings=[
                    ("rgb/image_rect_color", RGB_TOPIC),
                    ("rgb/camera_info", INFO_TOPIC),
                    ("depth_registered/image_rect", DEPTH_TOPIC),
                    ("points", POINTS_TOPIC),
                ],
            ),
        ],
    )

    distance_node = Node(
        package="krsri_distance",
        executable="distance_node",
        name="distance_node",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "imgsz": LaunchConfiguration("imgsz"),
                "conf": LaunchConfiguration("conf"),
            }
        ],
        remappings=[
            ("rgb/image_raw", RGB_TOPIC),
            ("depth/image_raw", DEPTH_TOPIC),
            ("camera_info", INFO_TOPIC),
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription(
        args + [kinect_node, point_cloud_container, distance_node, rviz]
    )
