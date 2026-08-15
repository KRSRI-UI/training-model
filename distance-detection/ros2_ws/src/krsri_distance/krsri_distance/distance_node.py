"""
Node inti: gabungkan deteksi YOLO 2D dengan citra depth Kinect menjadi posisi 3D.

Alur per pasang frame:
    RGB + depth (tersinkron) → YOLO → bbox 2D
                             → sampel depth di dalam bbox → jarak Z
                             → deproyeksi pinhole         → (X, Y, Z)
                             → terbitkan dua hal terpisah

Dua keluaran, sengaja dipisah:

  ~/detections_3d  (vision_msgs/Detection3DArray)
      Data mentah untuk dikonsumsi mesin — inilah yang di-subscribe node kontrol
      robot nanti. HANYA berisi deteksi dengan jarak yang benar-benar terukur;
      deteksi tanpa depth valid TIDAK dimasukkan, supaya node hilir tidak pernah
      menerima angka tebakan.

  ~/markers  (visualization_msgs/MarkerArray)
      Khusus RViz. Berisi SEMUA deteksi, termasuk yang tanpa depth — yang itu
      digambar abu-abu bertuliskan status ("TERLALU DEKAT" / "DEPTH TIDAK
      TERBACA"), agar operator melihat "terdeteksi tapi jarak tidak terbaca",
      bukan sekadar objeknya hilang dari layar.

Pemisahan ini yang membuat paket ini bisa dipakai untuk demo sekarang dan
disambung ke kontrol robot nanti tanpa ditulis ulang.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import message_filters
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from krsri_distance.detector import BonekaDetector, Detection
from krsri_distance.distance_estimator import (
    DEFAULT_CENTER_FRACTION,
    DEFAULT_MIN_VALID_RATIO,
    KINECT_MAX_RANGE_M,
    KINECT_MIN_RANGE_M,
    CameraIntrinsics,
    DistanceResult,
    DistanceStatus,
    estimate_distance,
)

MARKER_LIFETIME_NS = 300_000_000  # marker hilang sendiri kalau node berhenti
SPHERE_MIN_SIZE_M = 0.05
SPHERE_MAX_SIZE_M = 0.40
TEXT_HEIGHT_M = 0.06
TEXT_OFFSET_M = 0.12

#: Warna per nama kelas (RGBA). Kelas tak dikenal → putih.
CLASS_COLORS = {
    "boneka-asli": (0.0, 1.0, 0.0, 0.9),
    "boneka-dummy": (1.0, 0.0, 0.0, 0.9),
}
UNKNOWN_COLOR = (1.0, 1.0, 1.0, 0.9)
UNUSABLE_COLOR = (0.6, 0.6, 0.6, 0.6)

STATUS_LABELS = {
    DistanceStatus.TOO_CLOSE: "TERLALU DEKAT (<0.8m)",
    DistanceStatus.TOO_FAR: "TERLALU JAUH (>4m)",
    DistanceStatus.NO_DATA: "DEPTH TIDAK TERBACA",
}


class DistanceNode(Node):
    """Menyatukan deteksi 2D dan citra depth menjadi deteksi 3D."""

    def __init__(self) -> None:
        super().__init__("distance_node")
        self._declare_parameters()

        self._bridge = CvBridge()
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._detector = self._build_detector()

        self._detections_pub = self.create_publisher(
            Detection3DArray, "detections_3d", 10
        )
        self._marker_pub = self.create_publisher(MarkerArray, "markers", 10)
        self._debug_pub = self.create_publisher(Image, "debug_image", 10)

        self.create_subscription(
            CameraInfo, "camera_info", self._on_camera_info, qos_profile_sensor_data
        )

        rgb_sub = message_filters.Subscriber(
            self, Image, "rgb/image_raw", qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, "depth/image_raw", qos_profile=qos_profile_sensor_data
        )
        # Timestamp RGB dan depth tidak pernah identik pada kamera nyata, jadi
        # penyelarasan harus approximate, bukan exact.
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self._sync.registerCallback(self._on_frames)

        self.get_logger().info("distance_node siap — menunggu camera_info dan frame.")

    # ── Setup ───────────────────────────────────────────────────────────────

    def _declare_parameters(self) -> None:
        self.declare_parameter("model_path", "")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.50)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("stabilize", True)
        self.declare_parameter("min_hits", 3)
        self.declare_parameter("center_fraction", DEFAULT_CENTER_FRACTION)
        self.declare_parameter("min_valid_ratio", DEFAULT_MIN_VALID_RATIO)
        self.declare_parameter("min_range_m", KINECT_MIN_RANGE_M)
        self.declare_parameter("max_range_m", KINECT_MAX_RANGE_M)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("sync_queue_size", 5)

    def _build_detector(self) -> BonekaDetector:
        model_path = str(self.get_parameter("model_path").value)
        if not model_path:
            self.get_logger().fatal(
                "Parameter `model_path` kosong. Contoh:\n"
                "    ros2 run krsri_distance distance_node "
                "--ros-args -p model_path:=/path/ke/best_ncnn_model"
            )
            raise SystemExit(1)

        try:
            detector = BonekaDetector(
                model_path=model_path,
                imgsz=int(self.get_parameter("imgsz").value),
                conf=float(self.get_parameter("conf").value),
                iou=float(self.get_parameter("iou").value),
                stabilize=bool(self.get_parameter("stabilize").value),
                min_hits=int(self.get_parameter("min_hits").value),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().fatal(f"Gagal memuat model: {exc}")
            raise SystemExit(1) from exc

        self.get_logger().info(
            f"Model dimuat: {model_path} — kelas {detector.class_names}"
        )
        return detector

    def _on_camera_info(self, msg: CameraInfo) -> None:
        """Ambil intrinsik sekali; camera_info dikirim berulang tapi isinya tetap."""
        if self._intrinsics is not None:
            return
        try:
            self._intrinsics = CameraIntrinsics.from_k_matrix(msg.k)
        except ValueError as exc:
            self.get_logger().error(f"camera_info tidak dapat dipakai: {exc}")
            return
        self.get_logger().info(
            f"Intrinsik diterima: fx={self._intrinsics.fx:.1f} "
            f"fy={self._intrinsics.fy:.1f} cx={self._intrinsics.cx:.1f} "
            f"cy={self._intrinsics.cy:.1f}"
        )

    # ── Loop utama ──────────────────────────────────────────────────────────

    def _on_frames(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self._intrinsics is None:
            self.get_logger().warn(
                "Belum menerima camera_info — tanpa intrinsik posisi 3D tidak bisa "
                "dihitung. Periksa remap topic camera_info.",
                throttle_duration_sec=5.0,
            )
            return

        try:
            frame_bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self._bridge.imgmsg_to_cv2(depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Konversi citra gagal: {exc}", throttle_duration_sec=5.0
            )
            return

        if depth.shape[:2] != frame_bgr.shape[:2]:
            self.get_logger().error(
                f"Ukuran depth {depth.shape[:2]} != RGB {frame_bgr.shape[:2]}. "
                "Depth belum diregistrasi ke RGB — jarak akan salah. "
                "Pastikan memakai topic depth_registered.",
                throttle_duration_sec=5.0,
            )
            return

        detections = self._detector.detect(frame_bgr)
        results = [self._measure(depth, det, depth_msg.encoding) for det in detections]

        self._publish_detections(rgb_msg, detections, results)
        if bool(self.get_parameter("publish_markers").value):
            self._publish_markers(rgb_msg, detections, results)
        if bool(self.get_parameter("publish_debug_image").value):
            self._publish_debug_image(rgb_msg, frame_bgr, detections, results)

        self._log_summary(detections, results)

    def _measure(
        self, depth: np.ndarray, det: Detection, encoding: str
    ) -> DistanceResult:
        return estimate_distance(
            depth_image=depth,
            bbox=det.bbox,
            intrinsics=self._intrinsics,
            encoding=encoding,
            center_fraction=float(self.get_parameter("center_fraction").value),
            min_valid_ratio=float(self.get_parameter("min_valid_ratio").value),
            min_range_m=float(self.get_parameter("min_range_m").value),
            max_range_m=float(self.get_parameter("max_range_m").value),
        )

    # ── Keluaran 1: data untuk mesin ────────────────────────────────────────

    def _publish_detections(
        self, rgb_msg: Image, detections: List[Detection], results: List[DistanceResult]
    ) -> None:
        """Terbitkan HANYA deteksi berjarak valid — node hilir tak boleh dapat tebakan."""
        array = Detection3DArray()
        array.header = rgb_msg.header

        for det, res in zip(detections, results):
            if not res.status.is_usable or res.position_m is None:
                continue

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det.class_name
            hypothesis.hypothesis.score = det.confidence
            hypothesis.pose.pose.position.x = res.position_m[0]
            hypothesis.pose.pose.position.y = res.position_m[1]
            hypothesis.pose.pose.position.z = res.position_m[2]
            hypothesis.pose.pose.orientation.w = 1.0

            detection = Detection3D()
            detection.header = rgb_msg.header
            detection.id = str(det.track_id) if det.track_id is not None else ""
            detection.results = [hypothesis]
            detection.bbox.center.position.x = res.position_m[0]
            detection.bbox.center.position.y = res.position_m[1]
            detection.bbox.center.position.z = res.position_m[2]
            detection.bbox.center.orientation.w = 1.0
            width_m, height_m = res.size_m if res.size_m else (0.0, 0.0)
            detection.bbox.size.x = width_m
            detection.bbox.size.y = height_m
            # Kedalaman objek tidak terukur dari satu sisi; pakai lebar sebagai
            # perkiraan wajar agar kotak 3D-nya tidak pipih tanpa arti.
            detection.bbox.size.z = width_m
            array.detections.append(detection)

        self._detections_pub.publish(array)

    # ── Keluaran 2: visualisasi RViz ────────────────────────────────────────

    def _publish_markers(
        self, rgb_msg: Image, detections: List[Detection], results: List[DistanceResult]
    ) -> None:
        """Gambar SEMUA deteksi; yang tanpa jarak valid ditandai abu-abu + statusnya."""
        markers = MarkerArray()
        lifetime = Duration(sec=0, nanosec=MARKER_LIFETIME_NS)

        for index, (det, res) in enumerate(zip(detections, results)):
            position = res.position_m
            if position is None:
                # Tanpa depth tidak ada posisi nyata. Letakkan penanda di jarak
                # nominal agar tetap terlihat, dan katakan terus terang di label.
                position = self._fallback_position(det)

            color = self._marker_color(det, res)
            markers.markers.append(
                self._make_sphere(rgb_msg, index, position, color, res, lifetime)
            )
            markers.markers.append(
                self._make_label(rgb_msg, index, position, color, det, res, lifetime)
            )

        self._marker_pub.publish(markers)

    def _fallback_position(self, det: Detection) -> Tuple[float, float, float]:
        """Posisi perkiraan pada jarak nominal, khusus deteksi tanpa depth valid."""
        nominal_z = float(self.get_parameter("min_range_m").value)
        u = (det.bbox[0] + det.bbox[2]) / 2.0
        v = (det.bbox[1] + det.bbox[3]) / 2.0
        x = (u - self._intrinsics.cx) * nominal_z / self._intrinsics.fx
        y = (v - self._intrinsics.cy) * nominal_z / self._intrinsics.fy
        return x, y, nominal_z

    def _marker_color(self, det: Detection, res: DistanceResult):
        if not res.status.is_usable:
            return UNUSABLE_COLOR
        return CLASS_COLORS.get(det.class_name, UNKNOWN_COLOR)

    def _make_sphere(self, rgb_msg, index, position, color, res, lifetime) -> Marker:
        marker = Marker()
        marker.header = rgb_msg.header
        marker.ns = "boneka"
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0

        size = SPHERE_MIN_SIZE_M
        if res.size_m is not None:
            size = min(max(min(res.size_m), SPHERE_MIN_SIZE_M), SPHERE_MAX_SIZE_M)
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.lifetime = lifetime
        return marker

    def _make_label(
        self, rgb_msg, index, position, color, det, res, lifetime
    ) -> Marker:
        marker = Marker()
        marker.header = rgb_msg.header
        marker.ns = "boneka_label"
        marker.id = index
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1] - TEXT_OFFSET_M
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        marker.scale.z = TEXT_HEIGHT_M
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.text = self._label_text(det, res)
        marker.lifetime = lifetime
        return marker

    def _label_text(self, det: Detection, res: DistanceResult) -> str:
        if res.status.is_usable and res.distance_m is not None:
            return f"{det.class_name} {res.distance_m:.2f} m ({det.confidence:.0%})"
        return f"{det.class_name} — {STATUS_LABELS.get(res.status, res.status.value)}"

    def _publish_debug_image(
        self,
        rgb_msg: Image,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        results: List[DistanceResult],
    ) -> None:
        """Citra 2D beranotasi — berguna untuk `rqt_image_view` dan rekaman demo."""
        import cv2

        annotated = frame_bgr.copy()
        for det, res in zip(detections, results):
            x1, y1, x2, y2 = det.bbox
            # OpenCV memakai BGR; CLASS_COLORS memakai RGB 0..1.
            r, g, b, _ = self._marker_color(det, res)
            color = (int(b * 255), int(g * 255), int(r * 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                self._label_text(det, res),
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        message = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        message.header = rgb_msg.header
        self._debug_pub.publish(message)

    def _log_summary(
        self, detections: List[Detection], results: List[DistanceResult]
    ) -> None:
        if not detections:
            return
        parts = [
            f"{det.class_name}="
            + (f"{res.distance_m:.2f}m" if res.status.is_usable else res.status.value)
            for det, res in zip(detections, results)
        ]
        self.get_logger().info(" | ".join(parts), throttle_duration_sec=1.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DistanceNode()
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
