# Laporan Proyek Machine Learning - Klasifikasi Audio Emosi - Eka Yulianto

## Domain Proyek

Dalam dunia teknologi modern, pengenalan emosi berdasarkan data audio merupakan bagian dari analisis sentimen multimodal yang dapat bermanfaat dalam berbagai aplikasi seperti asisten virtual, layanan pelanggan, terapi kesehatan mental, dan banyak lagi. Pengenalan emosi berbasis audio dapat meningkatkan interaksi pengguna dengan teknologi dengan memberikan respons yang lebih personal dan empatik.

Penelitian menunjukkan bahwa analisis emosi melalui sinyal audio merupakan pendekatan yang efektif dalam memahami keadaan psikologis seseorang, meski terdapat tantangan dalam perbedaan karakteristik suara seperti intonasi, nada, dan durasi yang harus diperhatikan dalam pengklasifikasian emosi. Proyek ini memanfaatkan teknik ekstraksi fitur audio seperti MFCC (Mel-Frequency Cepstral Coefficients) dan spektral untuk memperoleh representasi karakteristik suara, serta model machine learning dan deep learning untuk melakukan klasifikasi emosi secara akurat.

**Referensi:**
- [MFCC Based Audio Classification Using Machine Learning](https://www.researchgate.net/profile/Asha-Ashok-3/publication/355892482_MFCC_Based_Audio_Classification_Using_Machine_Learning/links/638163407b0e356feb848b3d/MFCC-Based-Audio-Classification-Using-Machine-Learning.pdf)
- [Automatic Classi¯cation of Bird Sounds: Using MFCC and Mel Spectrogram Features with Deep Learning](https://www.worldscientific.com/doi/pdf/10.1142/S2196888822500300)
- [Vimal, B., Surya, M., Sridhar, V. S., & Ashok, A. (2021, July). Mfcc based audio classification using machine learning. In 2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT) (pp. 1-4). IEEE.](https://www.sciencedirect.com/science/article/pii/S1877050920318512)
## Business Understanding

### Problem Statements

Proyek ini bertujuan menjawab beberapa permasalahan sebagai berikut:
- Bagaimana mengekstraksi fitur-fitur audio yang relevan untuk klasifikasi emosi?
- Algoritma mana yang lebih efektif antara Random Forest dan Deep Neural Network dalam klasifikasi emosi berbasis audio?
- Bagaimana memaksimalkan akurasi model dengan optimasi hyperparameter?

### Goals

Tujuan dari proyek ini adalah:
- Mengidentifikasi fitur audio yang efektif dalam pengklasifikasian emosi.
- Membandingkan performa model Random Forest dan Deep Neural Network dalam melakukan klasifikasi emosi.
- Mengoptimalkan performa model melalui teknik Bayesian Optimization untuk mencapai akurasi tertinggi.

### Solution Statements

1. Menggunakan dua algoritma klasifikasi, yaitu Random Forest dan Deep Neural Network, serta membandingkan kinerja keduanya berdasarkan metrik akurasi, f1-score, precision, dan recall.
2. Melakukan optimasi model menggunakan **Bayesian Optimization** pada model Deep Neural Network untuk mendapatkan kombinasi hyperparameter yang optimal.

## Data Understanding

Data yang digunakan dalam proyek ini merupakan dataset TESS (Toronto Emotional Speech Set) yang terdiri dari audio rekaman yang sudah dilabeli dengan kelas emosi tertentu. Setiap rekaman berisi informasi tentang intonasi dan nada suara yang dapat menggambarkan emosi seperti marah, sedih, bahagia, takut, jijik, netral, dan terkejut.
Jumlah data per-kelas emosi sebanyak 400 file dalam format `.WAV` (Waveform audio format) jadi total keseluruhan data sebanyak 2800 file `.WAV`. Jumlah data yang seimbang ini akan sangat berguna pada proses pelatihan sehingga tidak akan menimbulkan overfitting pada kelas-kelas data tertentu.

Data audio ini merupakan rekaman suara dari 2 orang wanita berusia 26 dan 64 tahun dengan mengucapkan sebuah kalimat `say the word ...` dengan intonasi dan nada yang bervariasi sehingga mewakili ketujuh emosi tersebut.  
Dataset ini dapat diunduh dari sumber berikut: [Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

### Variabel-Variabel dalam Dataset

Dataset audio biasanya tidak memiliki variabel seperti pada data tabular, tetapi terdapat label yang menunjukkan emosi yang direpresentasikan dalam rekaman. Setiap rekaman kemudian diekstrak fiturnya, sehingga variabel fitur yang digunakan adalah:

- **MFCC**: Mel-Frequency Cepstral Coefficients, yang mengukur amplitudo suara pada berbagai frekuensi.
- **Spektral Bandwidth**: Menggambarkan lebar spektrum dari sinyal.
- **Spektral Rolloff**: Frekuensi di bawah sejumlah persentase dari energi total spektrum.
- **Spektral Contrast**: Perbedaan frekuensi tertinggi dan terendah dari spektrum audio.

## Data Preparation

Tahapan data preparation yang dilakukan meliputi:
1. **Pengaturan Direktori dan DataFrame (file path dan label)** : Sebelum melakukan pemrosesan data, perlu diatur pengelompokkan file ke dalam direktori yang sesuai dengan jenis labelnya. kemudian membuat DataFrame awal berisi lokasi file beserta labelnya.
2. **Ekstraksi Fitur**: Setiap file audio diekstraksi menggunakan fungsi ekstraksi fitur seperti MFCC, spektral bandwidth, spektral rolloff, dan spektral contrast.
3. **Normalisasi Data**: Semua fitur dinormalisasi untuk meningkatkan performa model. Teknik normalisasi ini dilakukan agar rentang fitur tidak terlalu bervariasi, yang dapat mempersulit model dalam menemukan pola.
4. **Split Dataset**: Dataset dibagi menjadi data latih dan validasi untuk melatih dan menguji performa model.

## Modeling

Pada proyek ini, dua model diterapkan untuk klasifikasi emosi:

### Random Forest

Random Forest dipilih karena algoritma ini terkenal dalam klasifikasi data dengan noise.

Kelebihan dari Random Forest:
- Mudah untuk diimplementasikan dan sangat cocok untuk data dengan dimensi yang lebih sedikit.
- Kurang rentan terhadap overfitting pada data.

### Deep Neural Network (DNN)

Model DNN dirancang dengan beberapa lapisan tersembunyi, menggunakan fungsi aktivasi ReLU dan softmax pada lapisan output. Hyperparameter yang disesuaikan melalui Bayesian Optimization antara lain:
- Jumlah unit pada setiap lapisan tersembunyi.
- Learning rate untuk optimasi model.
- Batch size untuk proses pembelajaran.

Kelebihan dari DNN:
- Dapat menangkap pola yang lebih kompleks dalam data.
- Mendukung optimasi parameter yang memungkinkan hasil yang lebih presisi dibandingkan dengan model yang lebih sederhana.

### Hyperparameter Tuning dengan Bayesian Optimization

Bayesian Optimization digunakan untuk mencari kombinasi parameter terbaik pada DNN. Fungsi `def hyperparameter_tuning(params)` yang dioptimalkan adalah akurasi validasi, dengan beberapa parameter seperti jumlah unit pada setiap layer, learning rate, dan batch size.

## Evaluation

Proyek ini menggunakan beberapa metrik evaluasi untuk menilai performa model:

1. **Akurasi**: Proporsi prediksi benar terhadap semua prediksi.
2. **Precision**: Proporsi prediksi benar dari total prediksi positif.
3. **Recall**: Kemampuan model dalam menemukan semua sampel positif.
4. **F1-score**: Harmonik rata-rata precision dan recall, terutama berguna untuk mengatasi ketidakseimbangan data.

### Hasil Evaluasi

| Model          | Precision | Recall | F1-Score | Accuracy |
|----------------|-----------|--------|----------|----------|
| Random Forest  | 0.95      | 0.96   | 0.95     | 95%      |
| Neural Network | 0.95      | 0.95   | 0.95     | 95%      |

Dari hasil evaluasi, kedua model memberikan performa yang serupa dalam hal akurasi dan F1-score, dengan Random Forest sedikit lebih unggul dalam recall, sedangkan Neural Network lebih unggul dalam precision. 

Secara keseluruhan, kedua model memberikan performa yang tinggi, namun **Random Forest dipilih sebagai model terbaik** karena hasil recall-nya yang lebih baik dan stabil dalam klasifikasi emosi berbasis audio.

## Kesimpulan

Proyek ini berhasil mencapai tujuan utama, yaitu membangun model yang mampu mengklasifikasikan emosi berdasarkan data audio dengan akurasi yang memadai. Random Forest dan Neural Network masing-masing memiliki kelebihan dalam klasifikasi ini, dan dengan melakukan optimasi hyperparameter, performa model dapat ditingkatkan lebih lanjut.
