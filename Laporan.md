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

  








































































































Referensi:

World Gold Council. (2023). Gold Market Commentary

Zhang et al. (2020). Forecasting gold price with LSTM. Journal of Economic Forecasting.
