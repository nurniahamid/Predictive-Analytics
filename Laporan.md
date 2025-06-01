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
- Tautan file: https://www.investing.com/currencies/xau-usd-historical-data
- Jumlah data: 
Data berisi 1368 baris dalam data set
Data mempunyai 7 kolom, yaitu : Date, Price, Open, High, Low Vol, dan Change %

Kolom:
- Date: Tanggal perdagangan 
- Price: Harga penutupan emas pada tanggal tersebut 
- Open: Harga pembukaan pada saat pasar dibuka pada hari itu 
- High: Harga tertinggi emas yang dicapai padda hari itu 
- Low: Harga terendah emas dalam hari itu
- Vol: Volume transaksi
- Change %: Perubahan harian dalam persentase

Kondisi Data :
- Fitur yang Digunakan: Dari seluruh kolom yang tersedia, hanya fitur Date, Price, Harga, High, dan Low yang digunakan dalam analisis.
Kolom Volume (Vol) dan Change dihapus (drop) karena tidak relevan dengan tujuan pemodelan.
- Missing Value: Terdapat nilai kosong (missing) pada kolom Volume, namun karena kolom ini tidak digunakan, maka missing value tersebut diabaikan.
- Duplikat: Tidak ditemukan baris duplikat setelah dilakukan pemeriksaan data.
- Outlier: Outlier diperiksa menggunakan visualisasi plot dan perhitungan z-score pada kolom Price.
Outlier tidak dihapus, karena masih dianggap valid dalam konteks data keuangan (misalnya, lonjakan harga wajar terjadi pada periode tertentu).

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
Model ARIMA digunakan untuk memodelkan data deret waktu (time series) dengan memperhatikan pola autokorelasi, tren, dan fluktuasi acak. ARIMA memiliki tiga komponen utama:
- AutoRegressive (AR) – Menjelaskan bahwa nilai masa kini dapat diprediksi dari kombinasi nilai masa lalu. Sebagai contoh, AR(2) menyatakan bahwa nilai sekarang dipengaruhi oleh dua nilai sebelumnya.
- Integrated (I) – Proses differencing data sebanyak d kali agar menjadi stasioner, yaitu data tidak memiliki tren atau perubahan varians sepanjang waktu.
- Moving Average (MA) – Menggambarkan bahwa nilai saat ini juga dipengaruhi oleh kesalahan (residual) dari prediksi sebelumnya. Misalnya, MA(2) mempertimbangkan dua kesalahan sebelumnya.

Model ARIMA dilambangkan sebagai ARIMA(p, d, q), dengan:
p: lag dari AR,
d: jumlah differencing,
q: lag dari MA.

Proses Pemodelan :
1. Transformasi Time Series
Data harga emas diubah menjadi format deret waktu dengan index tanggal agar dapat dianalisis menggunakan model ARIMA.

2. Identifikasi Parameter (p,d,q)
Digunakan plot ACF dan PACF untuk mengamati pola autokorelasi dan partial autokorelasi dalam data, sebagai panduan untuk menentukan nilai parameter AR dan MA.

3. Eksperimen Model
Tiga konfigurasi diuji:
- ARIMA(1,1,1)
Hasil menunjukkan bahwa koefisien AR(1) tidak signifikan (p-value > 0.05). AIC = 9461.276. Model ini kurang akurat dalam menangkap tren harga emas.
- ARIMA(2,1,2)
Semua parameter signifikan dan nilai AIC terendah (9460.719). Namun, model ini terlalu sensitif terhadap data latih (indikasi overfitting) dan tidak cukup responsif terhadap lonjakan harga pada data uji.
- ARIMA(2,2,2)
Model ini menghasilkan prediksi yang lebih stabil, dan tren harga pada data uji lebih akurat meskipun nilai AIC sedikit lebih tinggi (9463.848). Ini menunjukkan trade-off antara efisiensi statistik dan kemampuan prediktif yang lebih baik.

Model akhir dipilih berdasarkan keseimbangan antara stabilitas prediksi dan performa evaluasi.

### LSTM (Long Short-Term Memory)
LSTM adalah arsitektur khusus dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah vanishing gradient dan mampu menangkap ketergantungan jangka panjang dalam data deret waktu.

Mekanisme LSTM
Setiap unit LSTM terdiri dari komponen berikut:
- Forget Gate: Memutuskan informasi mana yang akan dilupakan dari memori sebelumnya.
- Input Gate: Menentukan informasi baru mana yang akan disimpan ke dalam cell state.
- Cell State: Menyimpan memori jangka panjang yang diperbarui setiap waktu.
- Output Gate: Menentukan nilai output berdasarkan memori yang diperbarui.

Proses Pemodelan : 
1. Preprocessing Data
- Data harga emas diambil dari kolom Price, kemudian dinormalisasi menggunakan MinMaxScaler ke rentang [0,1].
- Data dibagi menjadi:
Training set: 80% (1108 titik data)
Testing set: 20% (278 titik data)

2. Pembuatan Window Time Series
- Teknik sliding window digunakan dengan window size = 30.
- Input (X): 30 hari historis
- Target (y): harga emas pada hari ke-31
Bentuk data:
- X_train: (1078, 30)
- y_train: (1078,)
- X_test: (248, 30)
- y_test: (248,)

3. Arsitektur Model
- Dibangun menggunakan Keras dengan arsitektur:
<pre>
```python
model = Sequential([
    Input(shape=(30, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])
</pre>
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- EarlyStopping: digunakan untuk menghentikan pelatihan lebih awal jika loss tidak membaik selama 5 epoch.
- Model ini mampu menangkap tren harga emas dengan baik dan menunjukkan performa generalisasi yang lebih stabil dibanding model ARIMA, terutama pada data uji.


## Evaluation
#### Evaluasi Model ARIMA 
Model ARIMA digunakan sebagai baseline dalam memprediksi harga emas harian. Berdasarkan hasil pelatihan dan pengujian:
- ARIMA(2,2,2)
    - MAE (Mean Absolute Error): 255.01
    - RMSE (Root Mean Squared Error): 333.80
Nilai error yang cukup tinggi menunjukkan bahwa model ARIMA memiliki keterbatasan dalam menangkap pola jangka pendek dari data harga emas.

LSTM:
Model LSTM diterapkan untuk menangkap pola urutan (sequential pattern) dalam data time series. Berdasarkan hasil pengujian:
- MAPE (Mean Absolute Percentage Error): 2.21%
- Akurasi Estimasi: 97.79%
Model LSTM menunjukkan performa yang jauh lebih baik dibanding ARIMA, terutama dalam menangkap tren jangka pendek dan fluktuasi data emas harian.

#### Keterkaitan dengan Business Understanding

Pertanyaan
1. Apakah masalah bisnis berhasil dijawab?
Ya, model berhasil memberikan prediksi harga emas harian yang akurat dan dapat digunakan sebagai dasar pengambilan keputusan.

2. Apakah semua goals tercapai?
Ya, seluruh tujuan proyek terpenuhi:
- Model baseline (ARIMA) dan advanced (LSTM) berhasil dibangun.
- Evaluasi performa dilakukan secara kuantitatif.
- Model terbaik berhasil ditentukan berdasarkan hasil uji.

3. Apakah solusi berdampak?
Ya, model LSTM memberikan:
- Akurasi tinggi,
- Kemampuan menangkap pola musiman/tren,
- Potensi implementasi untuk strategi trading atau investasi jangka pendek berbasis data historis.


Model LSTM terbukti efektif dalam memprediksi harga emas dengan tingkat error yang rendah (MAPE 2.21%) dan akurasi tinggi (97.79%). Model ini dapat digunakan oleh pelaku pasar sebagai alat bantu analisis untuk pengambilan keputusan yang lebih cerdas dan berbasis data.


Referensi:
[1] World Gold Council. (2023). Gold Market Commentary
[2] Zhang et al. (2020). Forecasting gold price with LSTM. Journal of Economic Forecasting.
