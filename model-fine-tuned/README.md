# KRSRI Boneka Detection

Model computer vision berbasis YOLOv8 untuk mendeteksi dan membedakan dua jenis boneka target (`boneka-asli` dan `boneka-dummy`) pada robot KRSRI, dioptimalkan untuk berjalan di Raspberry Pi 4.

## Ringkasan Proyek

- **Task:** Object detection, 2 kelas (`boneka-asli`, `boneka-dummy`)
- **Model:** YOLOv8s, fine-tuned dari pretrained COCO weights
- **Dataset:** 450 gambar dari [Roboflow Universe](https://universe.roboflow.com/boneka-krsri/krsri-fbpsv) (lisensi CC BY 4.0) — lihat [`data/`](data/)
- **Target deployment:** Raspberry Pi 4, format NCNN
- **Training:** Two-phase (frozen backbone → full fine-tune), dijalankan di Google Colab

## Hasil

| Metrik | Nilai (valid set internal) |
|---|---|
| mAP@50 | ~0.994 |
| mAP@50-95 | ~0.90 |
| Precision | ~0.99 |
| Recall | ~1.00 |

Grafik training curve, distribusi kelas, dan diagnosis IoU tersedia di [`results/`](results/).

**Penting: angka di atas tidak mencerminkan performa lapangan sebenarnya.** Lihat bagian [Keterbatasan](#keterbatasan-yang-diketahui) di bawah sebelum menggunakan model ini untuk keputusan apa pun yang bergantung pada akurasi tinggi.

## Keterbatasan yang Diketahui

Model ini divalidasi dua cara: (1) terhadap valid set internal dari dataset Roboflow, dan (2) terhadap video uji lapangan yang merekam robot KRSRI sungguhan berinteraksi dengan boneka. Hasil kedua metode ini **berbeda jauh**, dan validasi (2) yang lebih dipercaya sebagai gambaran performa nyata.

### 1. Dataset training tidak representatif terhadap kondisi lapangan

Seluruh 450 gambar training berasal dari **satu sesi foto**, satu hari (12 Januari 2024), dalam rentang waktu ±43 menit, dengan background dan pencahayaan yang homogen. Split train/valid asli dari Roboflow bersifat acak, bukan berdasarkan waktu — analisis menunjukkan 93% gambar di valid set punya "gambar kembar" (frame berselisih <5 detik) di train set. Ini membuat mAP awal (>99%) sangat optimis karena model dievaluasi terhadap kondisi yang nyaris identik dengan yang dipelajarinya, bukan kondisi baru.

Notebook ini menyertakan langkah **re-split kronologis** (memisahkan train/valid berdasarkan waktu pengambilan, bukan acak) untuk evaluasi yang lebih jujur — tapi bahkan setelah itu, seluruh dataset tetap berasal dari satu ruangan/sesi yang sama.

### 2. Model gagal mendeteksi boneka saat menempel/tertutup badan robot

Ini temuan paling signifikan dari pengujian video lapangan. Dataset training tidak memiliki satu pun contoh boneka yang menempel atau tertutup sebagian oleh badan robot — situasi yang justru paling mungkin terjadi saat robot benar-benar mengangkut/berinteraksi dengan boneka di kompetisi. Saat kondisi ini terjadi, model hampir selalu gagal mendeteksi boneka sama sekali (false negative), bahkan ketika boneka terlihat jelas secara visual.

### 3. Sesekali muncul false positive pada objek berwarna serupa

Robot KRSRI berwarna kuning-oranye — mirip dengan warna boneka target. Pengujian video menunjukkan model beberapa kali salah mendeteksi bagian badan robot sebagai boneka, dan dalam satu kasus, kaki manusia yang masuk frame juga terdeteksi sebagai `boneka-asli`.

### 4. Class confusion

Pada beberapa frame, satu objek yang sama mendapat dua label sekaligus (`boneka-asli` dan `boneka-dummy` tumpang tindih) — menunjukkan model belum sepenuhnya yakin membedakan dua kelas dalam kondisi tertentu.

### Kenapa ini tidak diperbaiki di skrip training

Sudah dicoba beberapa pendekatan untuk memitigasi masalah di atas melalui augmentasi (HSV yang lebih luas, dsb) — lihat riwayat commit dan komentar di notebook. Hasilnya netral: beberapa false positive berkurang di satu titik, tapi muncul false positive baru di titik lain, dan masalah utama (boneka tertutup robot) tidak membaik sama sekali. Ini karena masalahnya bersifat struktural pada data (skenario tersebut tidak pernah dicontohkan), bukan sesuatu yang bisa diperbaiki lewat tuning hyperparameter atau augmentasi. Solusi yang tepat adalah menambah dataset dengan contoh skenario boneka-di-robot — di luar cakupan repo ini karena batasan hanya menggunakan dataset Roboflow yang sudah ada.

## Struktur Repo

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── krsri_dataset_yolov8.zip       # Dataset training (450 gambar, format YOLOv8)
├── notebooks/
│   └── krsri_boneka_training.ipynb    # Training, evaluasi, dan export model
├── models/
│   ├── README.md
│   ├── best_yolov8s.pt                # Model PyTorch
│   ├── best_yolov8s.onnx              # Model ONNX
│   ├── data.yaml                      # Konfigurasi kelas & path yang dipakai saat training
│   └── best_ncnn_model/               # Model NCNN, untuk deployment Raspberry Pi
├── results/
│   ├── README.md
│   ├── training_curves.png
│   └── inference_test.png
└── scripts/
    ├── raspi_detect.py                 # Inferensi realtime di Raspberry Pi (kamera/video/gambar), dgn mode --stable
    ├── test_video_detect.py            # Evaluasi model terhadap video (deteksi mentah per-frame)
    └── track_video_detect.py           # Evaluasi video versi stabil (tracking + temporal voting, 2-pass)
```

## Cara Pakai

### 1. Training (Google Colab)

Buka `notebooks/krsri_boneka_training.ipynb` di Google Colab dengan GPU aktif. Upload `data/krsri_dataset_yolov8.zip` ke Colab (drag-drop ke panel Files di kiri), sesuaikan `ZIP_PATH` di cell extract dataset kalau perlu, lalu jalankan seluruh notebook secara berurutan.

Notebook mencakup:
- Analisis dataset & pengecekan class imbalance
- Diagnosis kebocoran temporal antara train/valid split
- Re-split kronologis untuk evaluasi yang lebih jujur
- Training two-phase (frozen backbone → full fine-tune)
- Evaluasi dengan/tanpa Test-Time Augmentation
- Diagnosis IoU dan confusion matrix
- Export ke NCNN dan ONNX untuk deployment

### 2. Deploy ke Raspberry Pi 4

```bash
pip install ultralytics opencv-python
python3 raspi_detect.py --source 0 --imgsz 640
```

> **Penting — `--imgsz` wajib 640.** Model NCNN di repo ini diekspor *fixed-shape* 640×640 (lihat `best_ncnn_model/metadata.yaml`). Menjalankannya di ukuran lain (mis. 416) membuat NCNN mengeluarkan **ratusan bounding box sampah** — sumber besar ketidakstabilan deteksi di lapangan. `raspi_detect.py` sekarang **menolak jalan** kalau `--imgsz` tidak cocok dengan ukuran ekspor. Kalau butuh 320/416 demi kecepatan di Raspi, **ekspor ulang** model di ukuran itu dari notebook, jangan paksa lewat flag.

Secara default `raspi_detect.py` kini juga mengaktifkan **mode stabilisasi** (`--stable`): tracking IoU ringan + voting kelas berjalan + konfirmasi min-hits, untuk meredam false-positive kedip dan label yang gonta-ganti. Pakai `--raw` untuk kembali ke deteksi mentah per-frame (perilaku lama).

`raspi_detect.py` membaca model NCNN dari folder `best_ncnn_model/` di direktori kerja yang sama — `scripts/raspi_detect.py` dan `models/best_ncnn_model/` perlu berada di folder yang sama saat dijalankan. Lihat docstring di `scripts/raspi_detect.py` untuk opsi lengkap (Pi Camera, mode headless, simpan output, dsb).

### 3. Menguji model terhadap video

```bash
pip install ultralytics opencv-python
python3 scripts/test_video_detect.py --video path/ke/video.mp4 --weights models/best_yolov8s.pt
```

Menghasilkan video baru dengan bounding box, label, dan confidence score untuk setiap deteksi, plus ringkasan statistik di terminal. Berguna untuk menguji model di luar dataset training — sangat direkomendasikan sebelum menggunakan model di kompetisi sungguhan.

**Versi stabil (offline, 2-pass):** untuk review video yang jauh lebih mulus, pakai `scripts/track_video_detect.py`. Skrip ini menjalankan object tracking (ByteTrack), lalu menetapkan satu kelas per objek lewat **temporal voting** (mayoritas ditimbang confidence sepanjang hidup objek), membuang track "kedip" yang hidup di bawah `--min-track-len` frame sebagai false positive, dan menghaluskan box. Pada video uji, ini menurunkan label flip antar-frame dari **6,1% → 0,1%**, objek berlabel-ganda dari **13,6% → 0,0%** frame, dan frekuensi jumlah-deteksi berubah dari **42,4% → 28,8%** — tanpa mengubah bobot model.

```bash
python3 scripts/track_video_detect.py --video path/ke/video.mp4 --weights models/best_yolov8s.pt
```

## Requirements

Lihat [`requirements.txt`](requirements.txt). Dependency utama: `ultralytics`, `opencv-python`.

## Pengembangan Selanjutnya

Berdasarkan keterbatasan yang diuraikan di atas, langkah paling berdampak untuk meningkatkan performa lapangan model ini adalah:

1. Menambahkan data training yang menunjukkan boneka menempel/tertutup sebagian oleh robot — skenario yang belum ada satu pun contohnya di dataset saat ini.
2. Menambahkan contoh negatif (robot tanpa boneka, objek lain berwarna serupa) agar model tidak salah memicu deteksi pada objek berwarna mirip.
3. Mengumpulkan data validasi dari sesi/lokasi yang berbeda dari sesi pengambilan dataset training, agar metrik evaluasi lebih mencerminkan kondisi lapangan sesungguhnya.
