import os
from glob import glob

from setuptools import find_packages, setup

package_name = "krsri_distance"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Owen Viriya",
    maintainer_email="owennnviriyaaa@gmail.com",
    description="Deteksi jarak boneka KRSRI dengan Kinect v1 + YOLOv8 untuk ROS 2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "kinect_node = krsri_distance.kinect_node:main",
            "distance_node = krsri_distance.distance_node:main",
        ],
    },
)
