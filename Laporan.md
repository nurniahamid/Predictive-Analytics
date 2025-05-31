# Laporan Proyek Machine Learning - Nurnia Hamid 

## Domain Proyek

Harga emas merupakan salah satu aset investasi utama yang sering dijadikan acuan oleh investor global karena kestabilannya dalam jangka panjang. Namun, volatilitas harga emas meningkat dalam beberapa tahun terakhir akibat ketidakpastian global, sebagaimana dilaporkan oleh World Gold Council (2023). Oleh karena itu, kemampuan untuk memprediksi harga emas secara akurat memiliki nilai strategis di bidang keuangan dan data science.

Proyek ini berfokus pada prediksi harga emas dengan menggunakan data historis tanpa memasukkan variabel eksternal (seperti inflasi atau nilai tukar), dengan alasan:
- Data historis tersedia dan terstruktur dengan baik
- Model time series seperti ARIMA dan LSTM efektif dalam menangkap pola masa lalu
- Pendekatan eksploratif univariat lebih sederhana untuk tujuan pembelajaran dan evaluasi performa model

## Business Understanding
### Problem Statements
Bagaimana memprediksi harga emas untuk periode waktu mendatang berdasarkan data historis 2020–2025?
### Goals 
- Membangun model prediksi harga emas harian.
- Membandingkan performa antara model statistik (ARIMA) dan deep learning (LSTM).
- Menentukan model terbaik berdasarkan akurasi prediksi.
### Solution Statement
Dua pendekatan yang digunakan:
1. ARIMA – Model statistik untuk pola linier dalam time series.
1. LSTM – Model deep learning berbasis RNN untuk menangkap pola non-linier jangka panjang.

Metrik evaluasi:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## Data Understanding

Dataset : 
- Sumber data: Investing.com
- Tautan file: XAU_USD Historical Data.csv
- Jumlah data: 1368 baris, 6 kolom

Kolom:
- Date: Tanggal transaksi
- Price: Harga penutupan
- Open, High, Low: Harga pembukaan, tertinggi, dan terendah per hari
- Change %: Persentase perubahan harga harian

Kondisi Data :
- Missing value: Tidak ditemukan missing value pada kolom utama
- Duplikat: Tidak ditemukan duplikat baris setelah inspeksi
- Outlier: Dicek secara visual melalui plot dan nilai z-score pada kolom Price. Outlier diabaikan karena cenderung valid dalam konteks keuangan.

Eksplorasi Visual Awal : 
- Visualisasi tren (Price) menggunakan scatter plot
- Analisis korelasi antar fitur dengan heatmap
- Pemeriksaan distribusi harga dengan histogram

## Data Preparation
### Langkah-langkah yang dilakukan 
1. Konversi kolom Date ke datetime format
2. Sortir data berdasarkan tanggal secara menaik (oldest to newest)
3. Reset index agar linier dengan urutan waktu
4. Pilih kolom target: Price
5. Drop fitur non-relevan: seperti Vol, Change % yang tidak digunakan dalam pemodelan
6. Pembersihan angka: Hapus simbol dan ubah kolom numerik ke float (Price, Open, High, Low)
7. Split dataset:
80% untuk pelatihan
20% untuk pengujian
8. Normalisasi data: MinMaxScaler digunakan sebelum input ke LSTM
9. Sliding window: Membentuk data sekuens 30 hari sebagai input mode

## Modeling
### ARIMA (AutoRegressive Integrated Moving Average)
- Model time series tradisional ARIMA(p,d,q)
- Pemilihan parameter melalui ACF & PACF
- Evaluasi berbagai kombinasi parameter:
   - ARIMA(1,1,1): hasil kurang memuaskan
   - ARIMA(2,1,2): nilai AIC rendah, tetapi overfitting
   - ARIMA(2,2,2): stabil dan prediktif

### LSTM (Long Short-Term Memory)
- Recurrent Neural Network untuk menangkap ketergantungan jangka panjang
- Input: 30 hari historis untuk memprediksi harga hari ke-31
- Arsitektur:
   - 2 layer LSTM (50 unit masing-masing)
   - 1 Dense output layer
   - Adam optimizer + MSE loss
   - EarlyStopping digunakan untuk menghindari overfitting


## Evaluation
#### Evaluasi Model ARIMA 
Hasil Evaluasi Model
- ARIMA(2,2,2):
- MAE: 255.01
- RMSE: 333.80

LSTM:
- MAPE: 3.30%
- Akurasi estimasi: 96.7%

#### Keterkaitan dengan Business Understanding

Pertanyaan
1. Apakah masalah bisnis berhasil dijawab?
Jawaban : Ya, model berhasil memberikan prediksi harga emas harian yang cukup akurat.
2. Apakah semua goals tercapai?
Jawaban : Ya, model dibangun, dievaluasi, dan model terbaik ditentukan.
3. Apakah solusi berdampak?
Jawaban : Ya, LSTM memberikan akurasi tinggi dan dapat digunakan sebagai referensi untuk strategi investasi jangka pendek.

Model LSTM berhasil menjawab kebutuhan prediksi harga emas dengan performa yang sangat baik, dan dapat membantu pelaku pasar dalam mengambil keputusan berbasis data historis.


Referensi:
[1] World Gold Council. (2023). Gold Market Commentary
[2] Zhang et al. (2020). Forecasting gold price with LSTM. Journal of Economic Forecasting.
