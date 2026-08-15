"""
Test untuk logika estimasi jarak.

Tidak butuh ROS, tidak butuh Kinect, tidak butuh YOLO — semua citra depth
dibuat sintetis dengan numpy. Jalankan di mana saja:

    pytest ros2_ws/src/krsri_distance/test -v
"""

import numpy as np
import pytest

from krsri_distance.distance_estimator import (
    ENCODING_M_FLOAT32,
    ENCODING_MM_UINT16,
    CameraIntrinsics,
    DistanceStatus,
    center_roi,
    depth_to_meters,
    deproject,
    estimate_distance,
)

# Intrinsik bawaan Kinect v1 pada 640x480 (angka pendekatan; nilai sebenarnya
# datang dari camera_info saat runtime).
KINECT_INTRINSICS = CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5)

IMG_H, IMG_W = 480, 640
WALL_MM = 3000  # latar belakang
DOLL_MM = 1420  # objek


def make_scene_with_wall_background() -> np.ndarray:
    """Tembok di 3.0 m dengan tambalan boneka 1.42 m di tengah bbox uji."""
    depth = np.full((IMG_H, IMG_W), WALL_MM, dtype=np.uint16)
    depth[225:275, 325:375] = DOLL_MM
    return depth


# ── Inti: kenapa median-di-ROI-tengah, bukan cara lain ──────────────────────


def test_samples_object_not_background_corners():
    # Arrange — bbox 100x100 yang sudut-sudutnya berisi tembok
    depth = make_scene_with_wall_background()
    bbox = (300, 200, 400, 300)

    # Act
    result = estimate_distance(depth, bbox, KINECT_INTRINSICS)

    # Assert — ROI tengah 50% jatuh tepat di boneka, tembok terbuang
    assert result.status is DistanceStatus.OK
    assert result.distance_m == pytest.approx(1.42, abs=0.01)


def test_sampling_whole_bbox_would_return_background_distance():
    """Menjelaskan alasan center_fraction ada: sampel penuh membaca tembok."""
    # Arrange — boneka hanya 25% luas bbox, sisanya tembok
    depth = make_scene_with_wall_background()
    bbox = (300, 200, 400, 300)

    # Act
    result = estimate_distance(depth, bbox, KINECT_INTRINSICS, center_fraction=1.0)

    # Assert — median mayoritas = tembok, bukan boneka
    assert result.distance_m == pytest.approx(3.0, abs=0.01)


def test_ignores_invalid_zero_pixels_in_median():
    # Arrange — separuh ROI rusak (0 = "tidak terukur" pada Kinect)
    depth = np.full((IMG_H, IMG_W), DOLL_MM, dtype=np.uint16)
    depth[225:250, 325:375] = 0

    # Act
    result = estimate_distance(depth, (300, 200, 400, 300), KINECT_INTRINSICS)

    # Assert — nol tidak menarik median ke bawah
    assert result.status is DistanceStatus.OK
    assert result.distance_m == pytest.approx(1.42, abs=0.01)
    assert result.valid_ratio == pytest.approx(0.5, abs=0.01)


def test_ignores_nan_pixels_in_float_encoding():
    # Arrange — depth float meter dengan sebagian NaN
    depth = np.full((IMG_H, IMG_W), 1.42, dtype=np.float32)
    depth[225:250, 325:375] = np.nan

    # Act
    result = estimate_distance(
        depth, (300, 200, 400, 300), KINECT_INTRINSICS, encoding=ENCODING_M_FLOAT32
    )

    # Assert
    assert result.status is DistanceStatus.OK
    assert result.distance_m == pytest.approx(1.42, abs=0.01)


# ── Kejujuran saat data buruk ───────────────────────────────────────────────


def test_reports_no_data_when_depth_all_invalid():
    # Arrange — seluruh citra tidak terukur (mis. objek < 80 cm)
    depth = np.zeros((IMG_H, IMG_W), dtype=np.uint16)

    # Act
    result = estimate_distance(depth, (300, 200, 400, 300), KINECT_INTRINSICS)

    # Assert — tidak menebak, dan TIDAK mengembalikan 0.0 m
    assert result.status is DistanceStatus.NO_DATA
    assert result.distance_m is None
    assert result.position_m is None


def test_reports_no_data_when_valid_pixels_below_threshold():
    # Arrange — hanya 10% piksel valid, ambang default 20%
    depth = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    depth[225:230, 325:375] = DOLL_MM

    # Act
    result = estimate_distance(depth, (300, 200, 400, 300), KINECT_INTRINSICS)

    # Assert
    assert result.status is DistanceStatus.NO_DATA
    assert result.valid_ratio < 0.2


def test_flags_object_closer_than_kinect_minimum_range():
    # Arrange — 40 cm, di dalam blind zone Kinect v1
    depth = np.full((IMG_H, IMG_W), 400, dtype=np.uint16)

    # Act
    result = estimate_distance(depth, (300, 200, 400, 300), KINECT_INTRINSICS)

    # Assert — angkanya ada tapi ditandai tidak layak pakai
    assert result.status is DistanceStatus.TOO_CLOSE
    assert result.status.is_usable is False


def test_flags_object_beyond_kinect_maximum_range():
    # Arrange — 5 m, di luar jangkauan
    depth = np.full((IMG_H, IMG_W), 5000, dtype=np.uint16)

    # Act
    result = estimate_distance(depth, (300, 200, 400, 300), KINECT_INTRINSICS)

    # Assert
    assert result.status is DistanceStatus.TOO_FAR
    assert result.status.is_usable is False


def test_ok_status_is_the_only_usable_one():
    assert DistanceStatus.OK.is_usable is True
    assert DistanceStatus.NO_DATA.is_usable is False


# ── Geometri: deproyeksi piksel → titik 3D ──────────────────────────────────


def test_deprojects_optical_center_to_zero_lateral_offset():
    # Arrange / Act — piksel tepat di pusat optik
    x, y, z = deproject(319.5, 239.5, 2.0, KINECT_INTRINSICS)

    # Assert — lurus di depan kamera
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert z == pytest.approx(2.0)


def test_deprojects_pixel_offset_of_one_focal_length_to_z_metres_sideways():
    # Arrange — geser tepat sebesar fx piksel ke kanan
    u = KINECT_INTRINSICS.cx + KINECT_INTRINSICS.fx

    # Act
    x, _, _ = deproject(u, KINECT_INTRINSICS.cy, 2.0, KINECT_INTRINSICS)

    # Assert — pergeseran lateral = Z (sudut 45 derajat)
    assert x == pytest.approx(2.0)


def test_object_at_double_distance_deprojects_to_double_lateral_offset():
    # Arrange — piksel sama, jarak dua kali lipat
    near = deproject(400.0, 300.0, 1.0, KINECT_INTRINSICS)
    far = deproject(400.0, 300.0, 2.0, KINECT_INTRINSICS)

    # Assert — sinar yang sama, jarak tempuh dua kali
    assert far[0] == pytest.approx(near[0] * 2)
    assert far[1] == pytest.approx(near[1] * 2)


def test_estimates_3d_position_from_offset_bbox():
    # Arrange — boneka di kanan-bawah pusat optik, 2.0 m
    depth = np.full((IMG_H, IMG_W), 2000, dtype=np.uint16)
    bbox = (400, 300, 480, 380)  # pusat bbox = (440, 340)

    # Act
    result = estimate_distance(depth, bbox, KINECT_INTRINSICS)

    # Assert — X positif (kanan), Y positif (bawah), sesuai konvensi optical frame
    expected_x = (440 - 319.5) * 2.0 / 525.0
    expected_y = (340 - 239.5) * 2.0 / 525.0
    assert result.position_m[0] == pytest.approx(expected_x, abs=1e-3)
    assert result.position_m[1] == pytest.approx(expected_y, abs=1e-3)
    assert result.position_m[2] == pytest.approx(2.0, abs=1e-3)


def test_estimates_physical_size_from_bbox_and_distance():
    # Arrange — bbox 105x105 px pada 2.0 m; 105/525 * 2.0 = 0.40 m
    depth = np.full((IMG_H, IMG_W), 2000, dtype=np.uint16)

    # Act
    result = estimate_distance(depth, (300, 200, 405, 305), KINECT_INTRINSICS)

    # Assert
    assert result.size_m[0] == pytest.approx(0.40, abs=0.01)
    assert result.size_m[1] == pytest.approx(0.40, abs=0.01)


# ── Konversi encoding ───────────────────────────────────────────────────────


def test_converts_uint16_millimetres_to_metres():
    # Arrange / Act
    out = depth_to_meters(np.array([[1420, 2000]], dtype=np.uint16), ENCODING_MM_UINT16)

    # Assert
    assert out[0, 0] == pytest.approx(1.42)
    assert out[0, 1] == pytest.approx(2.0)


def test_passes_float32_metres_through_unchanged():
    # Arrange / Act
    out = depth_to_meters(np.array([[1.42]], dtype=np.float32), ENCODING_M_FLOAT32)

    # Assert
    assert out[0, 0] == pytest.approx(1.42)


def test_rejects_unknown_depth_encoding_loudly():
    # Salah menebak encoding = galat 1000x tanpa gejala. Harus gagal keras.
    with pytest.raises(ValueError, match="tidak dikenal"):
        depth_to_meters(np.zeros((2, 2), dtype=np.uint8), "8UC1")


# ── Penanganan bbox di tepi citra ───────────────────────────────────────────


def test_clips_bbox_that_extends_past_image_edge():
    # Arrange — bbox menembus tepi kanan-bawah
    depth = np.full((IMG_H, IMG_W), DOLL_MM, dtype=np.uint16)

    # Act
    result = estimate_distance(depth, (600, 440, 800, 600), KINECT_INTRINSICS)

    # Assert — tidak crash, tetap terukur dari bagian yang terlihat
    assert result.status is DistanceStatus.OK
    assert result.distance_m == pytest.approx(1.42, abs=0.01)


def test_reports_no_data_for_bbox_entirely_outside_image():
    # Arrange
    depth = np.full((IMG_H, IMG_W), DOLL_MM, dtype=np.uint16)

    # Act
    result = estimate_distance(depth, (700, 500, 800, 600), KINECT_INTRINSICS)

    # Assert
    assert result.status is DistanceStatus.NO_DATA


def test_tiny_bbox_falls_back_to_full_box_instead_of_collapsing():
    # Arrange — bbox 2x2 px; penyusutan 50% akan mengolapskannya jadi nol
    roi = center_roi((100, 100, 102, 102), IMG_W, IMG_H, fraction=0.5)

    # Assert — fallback ke bbox utuh, bukan None
    assert roi == (100, 100, 102, 102)


def test_center_roi_shrinks_large_box_to_middle_half():
    # Arrange / Act
    roi = center_roi((300, 200, 400, 300), IMG_W, IMG_H, fraction=0.5)

    # Assert
    assert roi == (325, 225, 375, 275)


def test_center_roi_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        center_roi((0, 0, 10, 10), IMG_W, IMG_H, fraction=0.0)


# ── Parsing camera_info ─────────────────────────────────────────────────────


def test_builds_intrinsics_from_camera_info_k_matrix():
    # Arrange — K row-major: [fx 0 cx, 0 fy cy, 0 0 1]
    k = [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]

    # Act
    intr = CameraIntrinsics.from_k_matrix(k)

    # Assert
    assert intr == KINECT_INTRINSICS


def test_rejects_uncalibrated_camera_info():
    # K semua nol = driver belum mengirim kalibrasi. Jangan diam-diam bagi nol.
    with pytest.raises(ValueError, match="terkalibrasi"):
        CameraIntrinsics.from_k_matrix([0.0] * 9)


def test_rejects_malformed_k_matrix():
    with pytest.raises(ValueError, match="9 elemen"):
        CameraIntrinsics.from_k_matrix([525.0, 0.0, 319.5])


def test_rejects_non_2d_depth_image():
    # Citra RGB (3 kanal) tidak sengaja dioper sebagai depth — harus gagal keras.
    with pytest.raises(ValueError, match="2D"):
        estimate_distance(
            np.zeros((IMG_H, IMG_W, 3), dtype=np.uint16),
            (300, 200, 400, 300),
            KINECT_INTRINSICS,
        )
