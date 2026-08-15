"""
Driver Kinect v1 (Xbox 360) untuk ROS 2 — pembungkus tipis di atas libfreenect.

ROS 2 tidak punya padanan resmi `freenect_stack` seperti ROS 1, jadi node ini
menyediakan yang minimum tapi benar:

  • Depth diminta dalam mode FREENECT_DEPTH_REGISTERED, yang membuat libfreenect
    MENYEJAJARKAN citra depth ke citra RGB di level driver. Ini krusial: kamera
    RGB dan kamera IR terpisah ~2,5 cm di badan Kinect, jadi tanpa registrasi
    piksel (u,v) pada RGB menunjuk objek yang BERBEDA dari piksel (u,v) pada
    depth — dan seluruh estimasi jarak jadi meleset tanpa gejala yang kelihatan.
  • Depth diterbitkan sebagai 16UC1 dalam MILIMETER (konvensi sensor_msgs).
  • RGB dan depth diberi timestamp yang sama persis, sehingga penyelarasan di
    hilir sepele dan tidak pernah gagal.

Point cloud TIDAK dibuat di sini — itu tugas `depth_image_proc`, paket resmi
ROS 2, yang menerima ketiga topic di bawah dan menerbitkan PointCloud2 berwarna.

Prasyarat:
    sudo apt install freenect python3-freenect
    sudo usermod -aG plugdev $USER        # lalu logout/login
"""

from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

KINECT_WIDTH = 640
KINECT_HEIGHT = 480
DEFAULT_FPS = 30.0

#: Intrinsik nominal Kinect v1 pada 640x480. Cukup untuk demo, tapi untuk
#: pengukuran serius kalibrasi sendiri dengan paket `camera_calibration`
#: lalu timpa lewat parameter fx/fy/cx/cy.
DEFAULT_FX = 525.0
DEFAULT_FY = 525.0
DEFAULT_CX = 319.5
DEFAULT_CY = 239.5

DEPTH_ENCODING = "16UC1"
RGB_ENCODING = "rgb8"
BYTES_PER_UINT16 = 2
CHANNELS_RGB = 3


class KinectNode(Node):
    """Menerbitkan RGB, depth teregistrasi, dan camera_info dari Kinect v1."""

    def __init__(self) -> None:
        super().__init__("kinect_node")

        self.declare_parameter("frame_id", "camera_rgb_optical_frame")
        self.declare_parameter("fps", DEFAULT_FPS)
        self.declare_parameter("fx", DEFAULT_FX)
        self.declare_parameter("fy", DEFAULT_FY)
        self.declare_parameter("cx", DEFAULT_CX)
        self.declare_parameter("cy", DEFAULT_CY)
        self.declare_parameter("device_index", 0)

        self._frame_id = self.get_parameter("frame_id").value
        self._device_index = int(self.get_parameter("device_index").value)
        fps = float(self.get_parameter("fps").value)

        self._freenect = self._import_freenect()
        self._camera_info = self._build_camera_info()

        self._rgb_pub = self.create_publisher(
            Image, "image_raw", qos_profile_sensor_data
        )
        self._depth_pub = self.create_publisher(
            Image, "depth_registered/image_raw", qos_profile_sensor_data
        )
        self._info_pub = self.create_publisher(
            CameraInfo, "camera_info", qos_profile_sensor_data
        )

        self._failure_count = 0
        self._timer = self.create_timer(1.0 / fps, self._publish_frame)
        self.get_logger().info(
            f"Kinect v1 aktif @ {fps:.0f} Hz, frame_id='{self._frame_id}', "
            f"depth = DEPTH_REGISTERED (sudah sejajar dengan RGB)"
        )

    def _import_freenect(self):
        """Import libfreenect dengan pesan galat yang menuntun, bukan traceback."""
        try:
            import freenect
        except ImportError as exc:
            self.get_logger().fatal(
                "Modul `freenect` tidak ditemukan. Pasang dengan:\n"
                "    sudo apt install freenect python3-freenect\n"
                "Kinect Xbox 360 juga WAJIB memakai adaptor daya 12V terpisah — "
                "USB saja tidak cukup untuk menyalakannya."
            )
            raise SystemExit(1) from exc

        if not hasattr(freenect, "DEPTH_REGISTERED"):
            self.get_logger().fatal(
                "libfreenect terpasang tapi tanpa DEPTH_REGISTERED. Versi terlalu "
                "lama — tanpa mode ini depth tidak sejajar dengan RGB dan seluruh "
                "estimasi jarak akan meleset."
            )
            raise SystemExit(1)
        return freenect

    def _build_camera_info(self) -> CameraInfo:
        """
        Susun CameraInfo dari parameter intrinsik.

        Karena depth sudah diregistrasi ke frame RGB, intrinsik RGB inilah yang
        berlaku untuk KEDUA citra — itu sebabnya hanya ada satu camera_info.
        """
        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)

        info = CameraInfo()
        info.header.frame_id = self._frame_id
        info.width = KINECT_WIDTH
        info.height = KINECT_HEIGHT
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _publish_frame(self) -> None:
        """Ambil sepasang frame dan terbitkan. Kegagalan dilaporkan, tidak ditelan."""
        try:
            rgb, _ = self._freenect.sync_get_video(self._device_index)
            depth, _ = self._freenect.sync_get_depth(
                self._device_index, self._freenect.DEPTH_REGISTERED
            )
        except Exception as exc:  # noqa: BLE001 — apa pun sebabnya, jangan diam
            self._failure_count += 1
            self.get_logger().error(
                f"Gagal membaca Kinect ({self._failure_count}x): {exc}",
                throttle_duration_sec=5.0,
            )
            return

        if rgb is None or depth is None:
            self._failure_count += 1
            self.get_logger().warn(
                "Kinect mengembalikan frame kosong — periksa adaptor 12V dan kabel USB.",
                throttle_duration_sec=5.0,
            )
            return

        self._failure_count = 0
        stamp = self.get_clock().now().to_msg()

        self._rgb_pub.publish(
            self._to_image_msg(rgb, RGB_ENCODING, KINECT_WIDTH * CHANNELS_RGB, stamp)
        )
        self._depth_pub.publish(
            self._to_image_msg(
                depth, DEPTH_ENCODING, KINECT_WIDTH * BYTES_PER_UINT16, stamp
            )
        )

        self._camera_info.header.stamp = stamp
        self._info_pub.publish(self._camera_info)

    def _to_image_msg(self, array, encoding: str, step: int, stamp) -> Image:
        """Bungkus array numpy jadi sensor_msgs/Image tanpa perantara cv_bridge."""
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = int(array.shape[0])
        msg.width = int(array.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = step
        msg.data = array.tobytes()
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = KinectNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        return
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
