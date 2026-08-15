# Panduan Setup — Ubuntu 22.04 + ROS 2 Humble + Kinect v1

Panduan lengkap dari laptop kosong sampai point cloud muncul di RViz.

**Ikuti berurutan.** Tiap tahap punya langkah verifikasi — jangan lanjut kalau
verifikasinya gagal, karena galat di tahap awal menyamar jadi galat aneh di tahap
akhir dan jauh lebih sulit dilacak.

| Tahap | Isi | Perkiraan waktu |
|---|---|---|
| [0](#tahap-0--kenapa-harus-ubuntu-2204) | Kenapa Ubuntu 22.04, bukan yang lain | baca 2 menit |
| [1](#tahap-1--install-ubuntu-2204) | Install Ubuntu 22.04 | 30–60 menit |
| [2](#tahap-2--install-ros-2-humble) | Install ROS 2 Humble | 20–40 menit |
| [3](#tahap-3--alat-build-colcon--rosdep) | Alat build (colcon, rosdep) | 5 menit |
| [4](#tahap-4--driver-kinect-v1-libfreenect) | Driver Kinect (libfreenect) | 15–30 menit |
| [5](#tahap-5--dependensi-ros-untuk-paket-ini) | Dependensi ROS paket ini | 10 menit |
| [6](#tahap-6--python--yolo) | Python + YOLO | 10–20 menit |
| [7](#tahap-7--build-dan-jalankan) | Build & jalankan | 10 menit |
| [8](#tahap-8--troubleshooting) | Troubleshooting | sesuai kebutuhan |

---

## Tahap 0 — Kenapa harus Ubuntu 22.04

Bukan preferensi, tapi keterikatan teknis berantai:

- **ROS 2 Humble** adalah rilis LTS yang secara resmi dibangun untuk **Ubuntu 22.04
  (Jammy)**. Di versi Ubuntu lain kamu harus build dari source — berjam-jam dan rawan gagal.
- **Kinect wajib Linux.** `libfreenect` tidak berjalan di Windows tanpa usaha besar,
  dan WSL2 bermasalah untuk perangkat USB berbandwidth tinggi seperti Kinect.
- Humble memakai **Python 3.10**, yang sudah jadi bawaan Ubuntu 22.04.

Kalau kamu memakai Ubuntu 24.04, ROS 2 Humble **tidak akan** terpasang dari apt —
kamu perlu Jazzy, dan dukungan Kinect v1 di sana lebih repot lagi.

> **Kalau kamu sudah punya Ubuntu 22.04**, langsung ke [Tahap 2](#tahap-2--install-ros-2-humble).

---

## Tahap 1 — Install Ubuntu 22.04

### Pilih cara install

| Cara | Kapan dipakai | Catatan |
|---|---|---|
| **Dual boot** | Direkomendasikan untuk proyek ini | Akses USB penuh, performa penuh. Kinect butuh keduanya. |
| **Laptop khusus** | Kalau ada laptop nganggur | Paling bersih, tanpa risiko ke Windows. |
| **VM (VirtualBox/VMware)** | ❌ Hindari | USB passthrough untuk Kinect sering gagal dan bandwidth-nya tidak cukup. |
| **WSL2** | ❌ Hindari | `usbipd` bisa mem-passthrough USB tapi tidak andal untuk Kinect. |

### Langkah dual boot

1. **Unduh ISO Ubuntu 22.04.5 LTS** dari <https://releases.ubuntu.com/22.04/>
   (pilih `ubuntu-22.04.x-desktop-amd64.iso`).

2. **Buat USB bootable** dengan [Rufus](https://rufus.ie/) (Windows) — pilih ISO,
   skema partisi GPT, target UEFI.

3. **Kecilkan partisi Windows** untuk memberi ruang Ubuntu. Di Windows, buka
   *Disk Management* → klik kanan drive C: → *Shrink Volume*. Sisakan **minimal 60 GB**
   (ROS 2 + PyTorch saja sudah belasan GB).

4. **Matikan Fast Startup di Windows** — kalau tidak, partisi Windows bisa terkunci
   dan Ubuntu tidak bisa membacanya:
   *Control Panel → Power Options → Choose what the power buttons do →
   Change settings that are currently unavailable →* hilangkan centang *Turn on fast startup*.

5. **Boot dari USB** (biasanya F12/F2/Del saat menyala, tergantung merek laptop),
   lalu pilih **Install Ubuntu alongside Windows Boot Manager**.

6. Saat instalasi, pilih **Normal installation** dan centang
   *Install third-party software for graphics and Wi-Fi hardware*.

### Verifikasi Tahap 1

```bash
lsb_release -a
```

Harus menampilkan `Description: Ubuntu 22.04.x LTS` dan `Codename: jammy`.
Kalau codename-nya bukan `jammy`, **berhenti di sini** — ROS 2 Humble tidak akan terpasang.

Lalu perbarui sistem:

```bash
sudo apt update && sudo apt upgrade -y
```

---

## Tahap 2 — Install ROS 2 Humble

### 2.1 Atur locale

ROS 2 mengharuskan locale UTF-8. Kalau tidak, sebagian tool gagal dengan pesan
galat yang sama sekali tidak menyinggung locale.

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

Verifikasi — output harus memuat `UTF-8`:

```bash
locale
```

### 2.2 Aktifkan repository universe

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
```

### 2.3 Tambahkan repository ROS 2

ROS mengubah cara distribusi kunci APT-nya beberapa waktu lalu. **Coba Cara A dulu**;
kalau gagal, pakai Cara B.

#### Cara A — paket `ros2-apt-source` (metode terbaru)

```bash
sudo apt install -y curl
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo apt install -y /tmp/ros2-apt-source.deb
```

#### Cara B — keyring manual (metode klasik)

```bash
sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

> Kalau keduanya gagal, cek panduan resmi terbaru di
> <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html> — langkah
> ini yang paling sering berubah dari waktu ke waktu.

### 2.4 Install ROS 2 Humble Desktop

```bash
sudo apt update
sudo apt install -y ros-humble-desktop
```

Unduhannya besar (~2 GB) dan lama. `ros-humble-desktop` sudah termasuk **RViz2**,
yang wajib untuk proyek ini — jangan pakai `ros-humble-ros-base` yang tanpa RViz.

### 2.5 Otomatis source setiap buka terminal

Tanpa ini, perintah `ros2` tidak akan dikenali di terminal baru:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Verifikasi Tahap 2

```bash
printenv ROS_DISTRO        # harus: humble
ros2 --help                # harus muncul daftar perintah
```

Uji komunikasi antar-node dengan dua terminal terpisah:

```bash
# Terminal 1
ros2 run demo_nodes_cpp talker

# Terminal 2
ros2 run demo_nodes_py listener
```

Listener harus mencetak `I heard: [Hello World: 1]`, dst. **Kalau ini gagal, jangan
lanjut** — semua tahap berikutnya bergantung pada komunikasi ROS yang sehat.

Tekan `Ctrl+C` di kedua terminal untuk berhenti.

---

## Tahap 3 — Alat build (colcon & rosdep)

```bash
sudo apt install -y ros-dev-tools python3-colcon-common-extensions python3-rosdep
```

Inisialisasi rosdep (sekali seumur hidup per mesin):

```bash
sudo rosdep init      # abaikan kalau bilang "already exists"
rosdep update
```

### Verifikasi Tahap 3

```bash
colcon version-check   # harus jalan tanpa "command not found"
```

---

## Tahap 4 — Driver Kinect v1 (libfreenect)

### ⚠️ Sebelum apa pun: adaptor daya

**Kinect Xbox 360 WAJIB memakai adaptor daya 12V terpisah.** Konektornya proprietary
dan USB saja **tidak cukup** untuk menyalakannya. Tanpa adaptor, perangkatnya tidak
akan pernah muncul di `lsusb` — dan kamu akan menghabiskan berjam-jam menyalahkan
driver padahal masalahnya listrik.

Yang kamu butuhkan: **Kinect Xbox 360 + AC adapter / USB breakout cable.**

### 4.1 Install libfreenect

```bash
sudo apt update
sudo apt install -y freenect libfreenect-dev python3-freenect
```

### 4.2 Izin akses USB

Secara bawaan hanya root yang boleh mengakses Kinect:

```bash
sudo usermod -aG plugdev $USER
```

**Logout lalu login kembali** agar keanggotaan grup berlaku. (Reboot juga bisa.)

Verifikasi setelah login ulang:

```bash
groups | grep plugdev     # harus muncul "plugdev"
```

Kalau setelah itu masih `permission denied`, tambahkan aturan udev manual:

```bash
sudo tee /etc/udev/rules.d/51-kinect.rules > /dev/null <<'RULES'
# Kinect Xbox 360 — audio, kamera, motor
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ad", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ae", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02b0", MODE="0666", GROUP="plugdev"
RULES
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 4.3 Blacklist modul kernel yang bentrok

Kernel Linux punya driver bawaan `gspca_kinect` yang **merebut** perangkat sebelum
libfreenect sempat memakainya. Gejalanya: Kinect terlihat di `lsusb` tapi
`freenect-glview` bilang tidak menemukan perangkat.

```bash
sudo modprobe -r gspca_kinect
echo "blacklist gspca_kinect" | sudo tee /etc/modprobe.d/blacklist-kinect.conf
```

### Verifikasi Tahap 4 — ini gerbang terpenting

Colok Kinect (**adaptor 12V terpasang**), lalu:

```bash
lsusb | grep -i microsoft
```

Harus muncul **tiga** baris (Xbox NUI Audio, Xbox NUI Camera, Xbox NUI Motor).
Kalau kosong → masalah daya atau kabel, bukan software.

Lalu uji tampilannya:

```bash
freenect-glview
```

**Harus muncul dua jendela: citra RGB dan citra depth berwarna-warni.**

Ini gerbang wajib. **Kalau `freenect-glview` gagal, JANGAN lanjut ke tahap
berikutnya** — semua sisanya bergantung pada driver yang berfungsi, dan kegagalan
di sini akan menyamar jadi galat ROS yang membingungkan nanti.

### 4.4 Verifikasi binding Python

Paket ini memanggil libfreenect dari Python, jadi binding-nya juga harus ada
**beserta mode `DEPTH_REGISTERED`**:

```bash
python3 -c "import freenect; print('DEPTH_REGISTERED' if hasattr(freenect,'DEPTH_REGISTERED') else 'TIDAK ADA — perlu build dari source')"
```

Kalau hasilnya `TIDAK ADA`, atau `import freenect` gagal, build dari source:

```bash
sudo apt install -y git cmake build-essential libusb-1.0-0-dev freeglut3-dev \
                    libxmu-dev libxi-dev python3-dev python3-numpy cython3

git clone https://github.com/OpenKinect/libfreenect.git ~/libfreenect
cd ~/libfreenect && mkdir -p build && cd build
cmake .. -DBUILD_PYTHON3=ON
make -j$(nproc)
sudo make install
sudo ldconfig

cd ~/libfreenect/wrappers/python
sudo python3 setup.py install
```

Lalu ulangi pengecekan `DEPTH_REGISTERED` di atas.

> **Kenapa `DEPTH_REGISTERED` wajib?** Kamera RGB dan kamera IR terpisah ~2,5 cm
> di badan Kinect. Tanpa mode ini, piksel `(u,v)` di citra RGB menunjuk objek yang
> **berbeda** dari piksel `(u,v)` di citra depth — dan seluruh estimasi jarak
> meleset **tanpa gejala yang terlihat**. Ini bug paling berbahaya di proyek
> semacam ini karena hasilnya tetap "kelihatan masuk akal".

---

## Tahap 5 — Dependensi ROS untuk paket ini

```bash
sudo apt install -y \
  ros-humble-vision-msgs \
  ros-humble-depth-image-proc \
  ros-humble-cv-bridge \
  ros-humble-message-filters \
  ros-humble-image-transport \
  ros-humble-rviz2 \
  ros-humble-rqt-image-view
```

Peran masing-masing:

| Paket | Untuk apa |
|---|---|
| `vision_msgs` | Tipe pesan `Detection3DArray` — keluaran untuk node kontrol robot |
| `depth_image_proc` | Membuat PointCloud2 berwarna dari depth + RGB. **Inilah yang menghasilkan awan titik di RViz** |
| `cv_bridge` | Konversi `sensor_msgs/Image` ↔ array OpenCV |
| `message_filters` | Menyelaraskan frame RGB dan depth yang timestamp-nya tidak identik |
| `rqt_image_view` | Melihat topic `debug_image` (citra 2D beranotasi) |

### Verifikasi Tahap 5

```bash
ros2 interface show vision_msgs/msg/Detection3DArray | head -5
```

Harus menampilkan definisi pesan, bukan galat "package not found".

---

## Tahap 6 — Python & YOLO

```bash
python3 -m pip install --upgrade pip
pip install --user "ultralytics>=8.4.0" "opencv-python>=4.8.0" ncnn
```

> **Unduhannya besar (~2–3 GB).** `ultralytics` menarik PyTorch sebagai dependensi
> meski kita memakai model NCNN. Siapkan koneksi dan ruang disk yang cukup.

### ⚠️ Peringatan konflik numpy

ROS 2 Humble memakai `python3-numpy` dari sistem. Sebagian versi `ultralytics`
menarik numpy yang lebih baru dan **membuat `cv_bridge` gagal** dengan galat seperti
`numpy.core.multiarray failed to import`.

Kalau itu terjadi, kembalikan numpy ke versi sistem:

```bash
pip install --user "numpy<2"
```

Jangan pakai virtualenv untuk proyek ini — paket Python ROS 2 dipasang ke
site-packages sistem dan venv membuat node ROS tidak menemukannya.

### Verifikasi Tahap 6

```bash
python3 -c "import ultralytics, cv2, numpy; print('ultralytics', ultralytics.__version__, '| numpy', numpy.__version__)"
python3 -c "from cv_bridge import CvBridge; print('cv_bridge OK')"
```

Keduanya harus jalan tanpa galat. Kalau `cv_bridge` gagal, itu gejala konflik numpy di atas.

---

## Tahap 7 — Build dan jalankan

### 7.1 Ambil kode

```bash
cd ~
git clone -b feat/kinect-distance-detection https://github.com/KRSRI-UI/training-model.git
cd training-model/distance-detection/ros2_ws
```

### 7.2 Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` membuat perubahan pada file Python langsung berlaku tanpa
build ulang — sangat menghemat waktu saat menyetel parameter.

> Setiap membuka terminal baru, kamu **wajib** menjalankan
> `source ~/training-model/distance-detection/ros2_ws/install/setup.bash` lagi.

### 7.3 Uji logika jarak tanpa hardware

```bash
cd src/krsri_distance && python3 -m pytest test -v && cd -
```

**26 test harus lolos.** Test ini tidak menyentuh Kinect maupun ROS, jadi kalau
lolos kamu tahu logika hitungannya benar — mempersempit pencarian bug nanti.

### 7.4 Siapkan model

Model NCNN ada di folder `model-fine-tuned/models/best_ncnn_model/` pada branch
`main`. Salin ke lokasi yang mudah dijangkau, misalnya `~/krsri/best_ncnn_model`.

### 7.5 Jalankan

```bash
ros2 launch krsri_distance krsri_distance.launch.py \
    model_path:=$HOME/krsri/best_ncnn_model
```

RViz akan terbuka otomatis dengan point cloud dan marker sudah terkonfigurasi.

### Verifikasi Tahap 7 — kriteria berhasil

1. **Point cloud muncul di RViz** — awan titik berwarna yang bisa diputar dengan mouse.
2. **Marker nempel di gumpalan titik berbentuk boneka.**
   Melayang di udara atau tembus tembok = ada yang salah.
3. **Validasi silang dengan meteran fisik** di 1 m, 2 m, 3 m. Kinect v1 wajar meleset
   beberapa sentimeter; meleset puluhan sentimeter berarti depth belum teregistrasi.
4. **Boneka di bawah 0,8 m harus muncul abu-abu** bertuliskan `TERLALU DEKAT`,
   **bukan** `0.00 m`.

Pemeriksaan lewat terminal:

```bash
ros2 topic list                                   # semua topic muncul?
ros2 topic hz /camera/depth_registered/points     # ~30 Hz?
ros2 topic echo /distance_node/detections_3d      # posisi 3D masuk akal?
ros2 run rqt_image_view rqt_image_view            # lihat /distance_node/debug_image
```

---

## Tahap 8 — Troubleshooting

### Kinect

**`lsusb` tidak menampilkan perangkat Microsoft**
Adaptor 12V tidak terpasang atau rusak. Ini penyebab nomor satu. Coba stopkontak
dan kabel USB lain.

**`freenect-glview`: "Could not open device"**
1. Sudah logout/login setelah `usermod -aG plugdev`? Cek dengan `groups`.
2. Modul bentrok masih aktif: `lsmod | grep gspca` — kalau ada, ulangi
   [Tahap 4.3](#43-blacklist-modul-kernel-yang-bentrok).
3. Coba sekali dengan `sudo freenect-glview`. Kalau ini berhasil sedangkan tanpa
   sudo gagal, masalahnya murni izin udev.

**Kinect terdeteksi tapi citranya membeku / putus-putus**
Kinect v1 memakan bandwidth USB besar. Colok ke port USB yang **tidak berbagi hub**
dengan perangkat lain, dan cabut webcam/hard disk eksternal saat pengujian.

### ROS 2

**`ros2: command not found`**
Terminal baru belum di-source: `source /opt/ros/humble/setup.bash`. Kalau sering
terjadi, pastikan barisnya benar-benar ada di `~/.bashrc`.

**`Package 'krsri_distance' not found`**
Workspace belum di-source: `source install/setup.bash` dari dalam `ros2_ws`.

**`colcon build` gagal karena dependensi**
```bash
cd ~/training-model/distance-detection/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

**Node jalan tapi tidak ada yang terjadi**
Cek koneksi topic-nya:
```bash
ros2 topic list
ros2 node info /distance_node
```
Kalau `/camera/image_raw` tidak ada, `kinect_node` yang bermasalah — periksa log-nya.

### RViz

**Point cloud tidak muncul**
1. *Fixed Frame* harus `camera_rgb_optical_frame` (di panel *Global Options*).
   Kalau tertulis merah, frame-nya salah ketik atau belum ada data.
2. *Reliability Policy* pada display PointCloud2 harus **Best Effort** — data sensor
   dikirim best-effort, dan RViz yang menuntut Reliable tidak akan menerima apa pun.
3. Pastikan datanya memang mengalir: `ros2 topic hz /camera/depth_registered/points`.

**Marker muncul tapi point cloud tidak (atau sebaliknya)**
Keduanya harus berada di frame yang sama. Cek `frame_id` yang diterbitkan
`kinect_node` cocok dengan *Fixed Frame* di RViz.

**Semua deteksi berlabel `DEPTH TIDAK TERBACA`**
1. Objeknya lebih dekat dari 0,8 m — mundurkan.
2. Permukaan objek menyerap IR (kain gelap/berbulu). Uji dulu dengan objek terang
   dan datar seperti kardus untuk memastikan pipeline-nya sehat.
3. Cek citra depth-nya benar-benar berisi:
   `ros2 topic echo /camera/depth_registered/image_raw --field header`

**Jarak terbaca tapi meleset jauh**
Kemungkinan besar depth tidak teregistrasi. Pastikan
[Tahap 4.4](#44-verifikasi-binding-python) melaporkan `DEPTH_REGISTERED` tersedia,
dan `distance_node` tidak mengeluarkan peringatan ukuran citra tidak sama.

### Python

**`numpy.core.multiarray failed to import`**
Konflik numpy antara pip dan sistem: `pip install --user "numpy<2"`.

**`ModuleNotFoundError: No module named 'ultralytics'` padahal sudah di-install**
Node ROS memakai `python3` sistem. Pastikan install pakai `pip install --user`,
bukan di dalam virtualenv.

---

## Ringkasan perintah verifikasi

Jalankan berurutan; tiap baris harus lolos sebelum lanjut:

```bash
lsb_release -a | grep jammy                       # Tahap 1
printenv ROS_DISTRO                               # Tahap 2 → humble
colcon version-check                              # Tahap 3
lsusb | grep -i microsoft                         # Tahap 4 → 3 baris
freenect-glview                                   # Tahap 4 → 2 jendela
python3 -c "import freenect; print(hasattr(freenect,'DEPTH_REGISTERED'))"
ros2 interface show vision_msgs/msg/Detection3DArray | head -3   # Tahap 5
python3 -c "from cv_bridge import CvBridge; print('OK')"          # Tahap 6
python3 -m pytest src/krsri_distance/test -q      # Tahap 7 → 26 passed
```

---

## Selanjutnya

Setelah semuanya jalan, baca [`README.md`](README.md) untuk penjelasan cara kerja
sistem, arti tiap topic, dan batasan yang perlu kamu sampaikan ke pembimbing —
terutama soal blind zone 0,8 m Kinect yang bertumpuk dengan keterbatasan model
saat boneka menempel di badan robot.
