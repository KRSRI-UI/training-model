# Dataset

`krsri_dataset_yolov8.zip` — 450 gambar boneka (`boneka-asli`, `boneka-dummy`) dalam format YOLOv8, diekspor dari Roboflow.

## Sumber & Lisensi

- **Sumber:** [Roboflow Universe — krsri (versi 2 train val)](https://universe.roboflow.com/boneka-krsri/krsri-fbpsv)
- **Lisensi:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — boleh digunakan ulang, dimodifikasi, dan didistribusikan ulang, **wajib mencantumkan atribusi** ke sumber di atas.
- Diekspor dari Roboflow pada 11 April 2024.

## Struktur setelah di-extract

```
krsri_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

## Catatan penting: keterbatasan dataset

Seluruh 450 gambar berasal dari **satu sesi foto**, satu hari (12 Januari 2024), rentang waktu ±43 menit, dengan background dan pencahayaan homogen. Split `train`/`valid` bawaan dari Roboflow bersifat **acak**, bukan berdasarkan waktu — ini menyebabkan kebocoran temporal (>90% gambar valid punya "gambar kembar" di train, berselisih <5 detik) yang membuat mAP awal terlihat sangat optimis (>99%).

Notebook training (`../notebooks/krsri_boneka_training.ipynb`) menyertakan langkah **re-split kronologis** yang meregenerasi ulang split train/valid berdasarkan waktu pengambilan foto (bukan acak), untuk evaluasi yang lebih jujur. Split hasil re-split ini **tidak disimpan sebagai file terpisah di repo** — dihasilkan otomatis dan deterministik setiap kali notebook dijalankan (Cell 4.6), langsung dari isi zip ini, jadi tidak perlu disimpan dua kali.

Lihat bagian "Keterbatasan yang Diketahui" di README utama repo untuk detail lebih lanjut soal implikasi keterbatasan dataset ini terhadap performa model di lapangan.
