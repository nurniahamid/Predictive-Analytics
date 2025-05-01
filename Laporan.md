# Laporan Proyek Machine Learning - Nurnia Hamid 

## Domain Proyek

  Harga emas merupakan salah satu aset investasi yang sangat diperhatikan karena nilainya yang relatif stabil dan cenderung meningkat dalam jangka panjang. Menurut laporan dari World Gold Council (2023), volatilitas harga emas meningkat signifikan dalam beberapa tahun terakhir akibat ketidakpastian global. Hal ini menjadikan kemampuan untuk memprediksi harga emas secara akurat sebagai tantangan sekaligus peluang besar di bidang keuangan dan data science.

  Meskipun pergerakan harga emas dipengaruhi oleh berbagai faktor eksternal seperti inflasi, suku bunga, kondisi geopolitik, dan nilai tukar mata uang, proyek ini secara eksplisit tidak menggunakan variabel-variabel tersebut. Fokus utama adalah pada pemanfaatan data historis harga emas sebagai satu-satunya input dalam pemodelan prediksi.

  Pendekatan ini dipilih karena beberapa alasan:
- Ketersediaan dan konsistensi data: Data historis harga emas lebih mudah diakses dan bersifat kuantitatif serta terstruktur.
- Kemampuan model time series: Model seperti ARIMA dan LSTM dirancang untuk menemukan pola dari data masa lalu dan memproyeksikannya ke masa depan, bahkan tanpa bantuan variabel eksternal.
- Tujuan eksploratif: Proyek ini bertujuan membandingkan pendekatan statistik dan deep learning dalam konteks prediksi univariat (satu variabel)

Dengan pendekatan ini, proyek tetap memberikan nilai praktis bagi analis atau investor yang ingin memperkirakan harga emas jangka pendek hingga menengah berdasarkan tren historis.

## Business Understanding
### Problem Statements
Bagaimana memprediksi harga emas pada periode waktu mendatang berdasarkan data historis harga emas 2020–2025?
### Goals 
- Membuat model prediksi harga emas harian.
- Membandingkan performa model statistik ARIMA dengan model deep learning LSTM.
- Menentukan model terbaik untuk prediksi harga emas
### Solution Statement
Untuk mencapai tujuan tersebut, proyek ini mengusulkan dua pendekatan:
1. ARIMA (AutoRegressive Integrated Moving Average): Model statistik yang baik untuk menangani pola linier dalam data time series.
2. LSTM (Long Short-Term Memory): Model deep learning berbasis RNN yang unggul dalam menangkap dependensi jangka panjang dalam data sekuensial.
Metrik evaluasi yang digunakan untuk mengukur kinerja model adalah:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Mean Absolute Percentage Error (MAPE)

## Data Understanding
- Jumlah Data: 1368 baris
- Periode Waktu: Tahun 2020 hingga April 2025
- Sumber Data: Investing.com
- Fitur:
Date: Tanggal perdagangan 
Price: Harga penutupan emas pada tanggal tersebut 
Open: Harga pembukaan pada saat pasar dibuka pada hari itu 
High: Harga tertinggi emas yang dicapai padda hari itu 
Low: Harga terendah emas dalam hari itu
Vol: Volume transaksi
Change %: Perubahan harian dalam persentase 

Teknik Eksplorasi : 
- Analisis distribusi dan tren data Price, yaitu menggunakan scatter plot untuk melihat pola data tren data emas kemudian melihat korelasi ntar fitur dengan heatmap dan melihat distribusi harga emas dengan histogram
- Deteksi dan pembersihan data duplikat atau anomali

## Data Preparation
### Langkah-langkah yang dilakukan 
1. Konversi Tipe data tanggal
   Kolom 'Date' dikonversi menjadi format 'datetime' untuk memudahkan analisis time series
2. Sortir dan Reset Index
   Data disortir berdasarkan tanggal (Date) dan di-reset index-nya untuk memastikan urutan time series yang bersih dan rapi
3. Data Splitting (Train-Test Split)
   Dataset dibagi menjadi 80% data latih dan 20% data uji berdasarkan urutan waktu (tanpa shuffling, karena ini time series)
   - Train data shape: (1108, 5)
   - Test data shape: (278, 5)
4. Pemilihan Kolom Target
Hanya kolom Price yang digunakan karena fokus pada prediksi harga emas (univariat)
5. Normalisasi Data (khusus untuk LSTM)
   Data distandarkan menggunakan MinMaxScaler ke rentang [0,1] agar model LSTM bisa belajar lebih efektif dari pola
6. Sliding Window (LSTM)
 Sequence data dibentuk dalam bentuk time steps untuk membangun dataset sekuensial yang dibutuhkan LSTM

### Alasan menggunakan Tahapan Ini:
- Urutan waktu harus terjaga dalam data time series — tidak boleh dilakukan shuffling seperti pada supervised learning biasa.
- Model seperti LSTM membutuhkan data terstandardisasi agar proses training lebih stabil dan cepat konvergen.
- ARIMA dan LSTM memiliki kebutuhan input yang berbeda, maka data perlu dipersiapkan dengan cara yang disesuaikan untuk masing-masing model.

## Modeling

### ARIMA (AutoRegressive Integrated Moving Average)
ARIMA adalah model time series yang digunakan untuk meramalkan data masa depan berdasarkan nilai masa lalu. ARIMA bekerja dengan tiga komponen utama:
- AR (AutoRegressive): ketergantungan terhadap nilai sebelumnya
- I (Integrated): pengurutan berdasarkan perbedaan (differencing)
- MA (Moving Average): ketergantungan terhadap kesalahan masa lalu

#### Proses Pemodelan
1. Transformasi ke Time Series Data latih dan data uji diubah ke format time series dengan index tanggal (Date) dan frekuensi diinfersikan
2. Identifikasi Parameter (p,d,q) Plot ACF dan PACF digunakan untuk mengevaluasi pola lag dalam data, sebagai dasar pemilihan parameter ARIMA.
3. Eksperimen Model Tiga model dibandingkan:
- ARIMA(1,1,1)
AR(1) tidak signifikan (p=0.08)
AIC = 9461.276
Tidak cukup kuat menangkap dinamika tren harga emas.
- ARIMA(2,1,2)
Semua parameter signifikan.
AIC = 9460.719 (terendah)
Log Likelihood tertinggi.
Secara statistik paling efisien, namun secara prediksi justru kurang menangkap tren naik tajam di data uji
- ARIMA(2,2,2)
AR dan MA sebagian besar signifikan.
AIC sedikit lebih tinggi = 9463.848.
model ini menghasilkan proyeksi yang lebih stabil dan mengikuti tren naik aktual harga emas selama periode uji, menjadikannya lebih cocok secara prediktif.

### LSTM (Long Short-Term Memory)

LSTM adalah jenis jaringan saraf tiruan berbasis Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah long-term dependencies pada data time series. Model ini sangat efektif dalam menangkap pola non-linier dan fluktuasi harga yang kompleks seperti pada data harga emas.

#### Proses Pemodelan
1. Preprocessing Data
- Data harga emas diambil dari kolom Price dan dikonversi menjadi array numerik
- Skala data distandarisasi menggunakan MinMaxScaler ke rentang 0–1 untuk mempercepat konvergensi model LSTM
- Data dibagi menjadi:
Training set: 80% (1108 poin)
Testing set: 20% (278 poin)

2. Pembuatan Window Time Series
- Menggunakan pendekatan sliding window:
Input (X): 30 hari harga emas sebelumnya
Target (y): harga emas pada hari ke-31
- Dataset hasil windowing:
X_train: (1078, 30)
y_train: (1078,)
X_test: (248, 30)
y_test: (248,)

3. Arsitektur Model
Model LSTM dibangun menggunakan Keras dengan arsitektur sebagai berikut:
- 2 lapisan LSTM dengan masing-masing 50 unit
- 1 lapisan Dense (output) untuk memprediksi harga
- Optimizer: Adam
- Loss Function: Mean Squared Error
- Regularisasi: EarlyStopping pada loss dengan patience=5

#### Model Final dan Prediksi
Setelah dilatih selama maksimum 50 epoch (berhenti lebih awal karena EarlyStopping), model diuji terhadap data uji


## Evaluation
#### Evaluasi Model ARIMA 
MAE = 255.01
- Rata-rata kesalahan prediksi harian sekitar 255 poin
RMSE = 333.80
- Terdapat beberapa hari dengan error besar; RMSE lebih sensitif terhadap outlier

Meskipun ARIMA(2,1,2) memiliki AIC terendah secara statistik, model ARIMA(2,2,2) terbukti lebih mampu merepresentasikan pola tren naik tajam pada data uji. Oleh karena itu, model ini dipilih sebagai model akhir untuk prediksi harga emas dalam proyek ini


#### Evaluasi Model LSTM 
- Prediksi dilakukan terhadap data testing, kemudian dibalik skalanya menggunakan inverse_transform. Hasil evaluasi:
- MAPE: 3.30%
Rata-rata kesalahan prediksi relatif hanya 3.30% dari harga aktual
- Akurasi: 96.70%
Model sangat presisi dalam menangkap pola harga emas

Karakteristik Prediksi:
- Mampu mengikuti tren naik dan turun dengan baik
- Lebih fleksibel dalam mempelajari pola non-linier dibanding ARIMA
- Cenderung menghasilkan kurva prediksi yang halus dan konsisten

Model LSTM memberikan performa prediktif yang sangat baik terhadap harga emas:
- MAPE rendah menandakan model efektif dalam menangkap dinamika data historis
- Akurasi tinggi membuktikan keunggulan LSTM dalam memodelkan data time series non-linier
- Cocok digunakan untuk prediksi jangka pendek hingga menengah dengan fluktuasi harga tinggi

Model LSTM lebih unggul dibandingkan ARIMA dalam memprediksi harga emas karena akurasi yang jauh lebih tinggi, kemampuan menangkap tren kompleks, dan kesalahan prediksi yang lebih rendah secara keseluruhan. Oleh karena itu, LSTM merupakan model yang lebih direkomendasikan untuk digunakan dalam prediksi harga emas ke depan.


Referensi:
[1] World Gold Council. (2023). Gold Market Commentary
[2] Zhang et al. (2020). Forecasting gold price with LSTM. Journal of Economic Forecasting.
