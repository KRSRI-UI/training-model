# KRSRI Distance Detection — Kinect v1 + YOLOv8 + ROS 2

Menambahkan **estimasi jarak 3D** di atas model deteksi boneka yang sudah ada di
[`../model-fine-tuned/`](../model-fine-tuned/), memakai sensor depth **Kinect Xbox 360**
dan divisualisasikan sebagai marker di atas point cloud **RViz**.

> **Hubungan dengan `model-fine-tuned/`**
> `model-fine-tuned/` = object detection saja (2D bounding box, tanpa jarak).
> Folder ini = deteksi **+ jarak**. Bobot model tidak diubah dan **tidak perlu
> di-training ulang** — folder ini hanya menambahkan lapisan pembacaan depth di atasnya.

## Konsep

Kamera biasa menghasilkan piksel berisi **warna**. Kinect menghasilkan citra kedua
dengan ukuran sama, tapi tiap pikselnya berisi **jarak dalam milimeter**. Keduanya
seperti dua lembar transparansi yang ditumpuk pas.

```
YOLO membaca lembar RGB  →  "boneka ada di kotak (300,200)-(400,300)"
                                        ↓
        tembus ke lembar depth, baca angka di kotak yang sama
                                        ↓
                    median piksel valid  →  1420 mm = 1,42 m
                                        ↓
      deproyeksi pinhole  →  (X, Y, Z)  →  Marker 3D di RViz
```

Jarak **tidak** dihitung dari ukuran bbox — itu metode kamera mono yang error-nya besar.
Kinect sudah menyediakan jaraknya per piksel; kita hanya membacanya dengan benar.

### Kenapa median dari bagian tengah bbox

- **Bukan piksel tengah saja** — piksel tunggal sering rusak (permukaan gelap menyerap
  IR, permukaan mengkilap memantulkannya menyamping, bayangan). Nilainya jadi `0`.
- **Bukan rata-rata seluruh bbox** — bbox itu persegi, boneka tidak. Sudut-sudutnya
  berisi tembok di belakang, dan rata-ratanya jadi tercampur latar.
- **Median dari ROI tengah 50%, setelah membuang piksel invalid** — tahan outlier dan
  memaksimalkan proporsi piksel objek.

Test `test_sampling_whole_bbox_would_return_background_distance` membuktikan ini dengan
angka: adegan yang sama menghasilkan **1,42 m** (ROI tengah) vs **3,00 m** (bbox penuh).

## Arsitektur

```
[Kinect v1] ──libfreenect(DEPTH_REGISTERED)──> [kinect_node]
                                                    │
                    ┌───────────────────────────────┼──────────────────────┐
                    │                               │                      │
             /camera/image_raw       /camera/depth_registered/    /camera/camera_info
                    │                       image_raw                     │
                    ├───────────────────────────────┴──────────────────────┤
                    │                                                      │
                    ↓                                                      ↓
          [depth_image_proc]                                      [distance_node]
                    │                                              │            │
   /camera/depth_registered/points              /distance_node/detections_3d    │
                    │                                    (untuk kontrol robot)  │
                    │                                                           │
                    └──────────────────> [RViz] <──────── /distance_node/markers
```

**Dua keluaran yang sengaja dipisah:**

| Topic | Tipe | Untuk |
|---|---|---|
| `~/detections_3d` | `vision_msgs/Detection3DArray` | Node kontrol robot. **Hanya** berisi deteksi dengan jarak terukur — node hilir tidak pernah menerima tebakan. |
| `~/markers` | `visualization_msgs/MarkerArray` | RViz. Berisi **semua** deteksi; yang tanpa depth valid digambar abu-abu bertuliskan statusnya. |
| `~/debug_image` | `sensor_msgs/Image` | Citra 2D beranotasi untuk `rqt_image_view` / rekaman demo. |

Karena terpisah, folder ini bisa dipakai untuk demo sekarang dan disambung ke kontrol
robot nanti tanpa ditulis ulang.

## Registrasi depth — jangan sampai kelewat

Kamera RGB dan kamera IR terpisah **~2,5 cm** di badan Kinect. Tanpa registrasi, piksel
`(u,v)` di RGB menunjuk objek yang **berbeda** dari piksel `(u,v)` di depth — dan seluruh
estimasi jarak meleset tanpa gejala yang kelihatan.

`kinect_node` meminta mode `FREENECT_DEPTH_REGISTERED`, sehingga libfreenect yang
menyejajarkan keduanya di level driver. `distance_node` juga menolak jalan kalau ukuran
citra depth dan RGB tidak sama, sebagai pengaman terakhir.

## Batasan yang harus kamu tahu

**Kinect v1 buta di bawah ~0,8 m.** Di jarak itu sensor mengembalikan `0`, bukan angka.

Ini bertumpuk dengan keterbatasan #2 pada [`../model-fine-tuned/README.md`](../model-fine-tuned/README.md):
model gagal mendeteksi boneka saat menempel/tertutup badan robot — dan pada saat itu
jaraknya hampir pasti di bawah 0,8 m. **Fitur jarak tidak menyelamatkan skenario itu.**

Yang dilakukan paket ini: melaporkannya secara jujur sebagai `NO_DATA` / `TOO_CLOSE`,
**bukan** mengembalikan `0.0 m` diam-diam. Robot yang mengira boneka berjarak 0 meter
akan mengambil keputusan berbahaya.

## Instalasi (Ubuntu 22.04 + ROS 2 Humble)

```bash
# 1. Dependensi ROS
sudo apt update
sudo apt install -y ros-humble-vision-msgs ros-humble-depth-image-proc \
                    ros-humble-cv-bridge ros-humble-message-filters ros-humble-rviz2

# 2. Driver Kinect v1
sudo apt install -y freenect python3-freenect
sudo usermod -aG plugdev $USER      # lalu logout/login

# 3. YOLO
pip install "ultralytics>=8.4.0" "opencv-python>=4.8.0" "numpy>=1.24.0"
```

> **Kinect Xbox 360 WAJIB memakai adaptor daya 12V terpisah.** USB saja tidak cukup —
> tanpa adaptor, perangkatnya tidak akan pernah terdeteksi.

### Verifikasi hardware sebelum apa pun

```bash
freenect-glview      # harus muncul dua jendela: RGB dan depth
```

Kalau ini gagal, jangan lanjut ke build — masalahnya di hardware/driver, bukan di kode.

## Build

```bash
cd distance-detection/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Menjalankan

```bash
ros2 launch krsri_distance krsri_distance.launch.py \
    model_path:=$HOME/path/ke/best_ncnn_model
```

RViz terbuka otomatis dengan point cloud + marker sudah terkonfigurasi.

Argumen lain: `imgsz` (default 640), `conf` (0.50), `use_rviz` (true), `frame_id`.

> **`imgsz` wajib cocok dengan ukuran ekspor model NCNN.** Model di repo ini diekspor
> fixed-shape 640×640; menjalankannya di ukuran lain membuat NCNN mengeluarkan ratusan
> box sampah. `BonekaDetector` membaca `metadata.yaml` dan **menolak jalan** kalau tidak cocok.

## Memverifikasi hasilnya benar

1. **Di RViz:** marker harus **nempel di gumpalan titik** yang bentuknya boneka.
   Melayang di udara atau tembus tembok = ada yang salah.
2. **Dengan meteran:** letakkan boneka di 1 m, 2 m, 3 m dan bandingkan dengan label.
   Kinect v1 wajar meleset beberapa sentimeter; kalau meleset puluhan sentimeter,
   kemungkinan besar depth belum teregistrasi.
3. **Di terminal:**
   ```bash
   ros2 topic echo /distance_node/detections_3d
   ros2 topic hz /camera/depth_registered/points
   ```

## Test

Logika inti (`distance_estimator.py`) sengaja murni numpy — tanpa ROS, tanpa YOLO,
tanpa Kinect. Bisa dijalankan di laptop mana pun, termasuk Windows:

```bash
cd ros2_ws/src/krsri_distance && python -m pytest test -v
```

26 test mencakup sampling median, penolakan piksel invalid, konversi encoding
(16UC1 mm ↔ 32FC1 meter), deproyeksi pinhole, bbox di tepi citra, dan penanganan
camera_info yang belum terkalibrasi.

Kalau angka jarak aneh di lapangan tapi test ini lolos, penyebabnya ada di lapisan
ROS/hardware — bukan di logika hitungannya. Ini mempersempit pencarian bug secara drastis.

## Struktur

```
distance-detection/
└── ros2_ws/src/krsri_distance/
    ├── krsri_distance/
    │   ├── distance_estimator.py   # Logika jarak murni (numpy saja) — teruji
    │   ├── detector.py             # Pembungkus YOLO, tanpa ROS
    │   ├── kinect_node.py          # Driver Kinect v1 → topic ROS 2
    │   └── distance_node.py        # Inti: sinkronisasi + estimasi + publikasi
    ├── launch/krsri_distance.launch.py
    ├── rviz/krsri.rviz
    └── test/test_distance_estimator.py
```

### Catatan: `detector.py` vs `raspi_detect.py`

`detector.py` adalah ekstraksi bersih dari `model-fine-tuned/scripts/raspi_detect.py`,
yang di sana menyatukan capture + inferensi + stabilisasi + rendering dalam satu loop
sehingga tidak bisa di-import node ROS. Tiga bug ikut diperbaiki:

1. **`min_hits` kini benar-benar berturut-turut.** Di versi asli `hits` tidak pernah
   di-reset saat track tidak ter-match, sehingga objek yang berkedip tiap 10 frame tetap
   lolos konfirmasi — padahal justru itu pola false-positive yang ingin dibuang.
2. **Nama kelas dibaca dari `model.names`**, bukan konstanta global yang bisa `IndexError`
   kalau bobot model diganti dengan jumlah kelas berbeda.
3. **`metadata.yaml` dibaca `yaml.safe_load`**, bukan parser baris manual; ketidakcocokan
   imgsz melempar exception alih-alih `sys.exit()`, supaya node ROS bisa melaporkannya.

`raspi_detect.py` sendiri sengaja **tidak diubah** agar folder `model-fine-tuned/` tetap
berdiri sendiri sebagai versi object-detection-saja.

## Pengembangan selanjutnya

- Kalibrasi intrinsik sendiri dengan paket `camera_calibration`, lalu timpa parameter
  `fx/fy/cx/cy` pada `kinect_node` — nilai bawaan (525, 525, 319.5, 239.5) hanya nominal.
- Publikasikan TF dari `camera_rgb_optical_frame` ke frame badan robot, agar posisi
  boneka bisa dinyatakan relatif terhadap robot, bukan relatif terhadap kamera.
- Sambungkan `~/detections_3d` ke node kontrol untuk navigasi menuju boneka.
