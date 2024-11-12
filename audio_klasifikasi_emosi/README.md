# Laporan Proyek Machine Learning - Klasifikasi Audio Emosi - Eka Yulianto

## Domain Proyek

Dalam dunia teknologi modern, pengenalan emosi berdasarkan data audio merupakan bagian dari analisis sentimen multimodal yang dapat bermanfaat dalam berbagai aplikasi seperti asisten virtual, layanan pelanggan, terapi kesehatan mental, dan banyak lagi. Pengenalan emosi berbasis audio dapat meningkatkan interaksi pengguna dengan teknologi dengan memberikan respons yang lebih personal dan empatik [1]. 

Penelitian menunjukkan bahwa analisis emosi melalui sinyal audio merupakan pendekatan yang efektif dalam memahami keadaan psikologis seseorang, meski terdapat tantangan dalam perbedaan karakteristik suara seperti intonasi, nada, dan durasi yang harus diperhatikan dalam pengklasifikasian emosi. Proyek ini memanfaatkan teknik ekstraksi fitur audio seperti MFCC (Mel-Frequency Cepstral Coefficients) dan spektral untuk memperoleh representasi karakteristik suara [3]. Model machine learning dan deep learning digunakan untuk melakukan pembelajaran terhadap data latih hingga menghasilkan prediksi yang akurat.

**Referensi:**
1. [Vimal, B., Surya, M., Sridhar, V. S., & Ashok, A. (2021, July). Mfcc based audio classification using machine learning. In 2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT) (pp. 1-4). IEEE.](https://www.researchgate.net/profile/Asha-Ashok-3/publication/355892482_MFCC_Based_Audio_Classification_Using_Machine_Learning/links/638163407b0e356feb848b3d/MFCC-Based-Audio-Classification-Using-Machine-Learning.pdf)
2. [Carvalho, S., & Gomes, E. F. (2023). Automatic classification of bird sounds: using MFCC and mel spectrogram features with deep learning. Vietnam Journal of Computer Science, 10(01), 39-54.](https://www.worldscientific.com/doi/pdf/10.1142/S2196888822500300)
3. [Vimal, B., Surya, M., Sridhar, V. S., & Ashok, A. (2021, July). Mfcc based audio classification using machine learning. In 2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT) (pp. 1-4). IEEE.](https://www.sciencedirect.com/science/article/pii/S1877050920318512)
## Business Understanding

### Problem Statements

Proyek ini bertujuan menjawab beberapa permasalahan sebagai berikut:
- Bagaimana mengekstraksi fitur-fitur audio yang relevan untuk klasifikasi emosi?
- Algoritma mana yang lebih efektif antara Random Forest dan Dense Neural Network dalam klasifikasi emosi berbasis audio?
- Bagaimana memaksimalkan akurasi model dengan optimasi hyperparameter?

### Goals

Tujuan dari proyek ini adalah:
- Mengidentifikasi fitur audio yang efektif dalam pengklasifikasian emosi.
- Membandingkan performa model Random Forest dan Dense Neural Network dalam melakukan klasifikasi emosi.
- Mengoptimalkan performa model melalui teknik Bayesian Optimization untuk mencapai akurasi tertinggi.

### Solution Statements

1. Menggunakan dua algoritma klasifikasi, yaitu Random Forest dan Dense Neural Network, serta membandingkan kinerja keduanya berdasarkan metrik akurasi, f1-score, precision, dan recall.
2. Melakukan optimasi model menggunakan **Bayesian Optimization** pada model Dense Neural Network untuk mendapatkan kombinasi hyperparameter yang optimal.

## Data Understanding

Data yang digunakan dalam proyek ini merupakan dataset TESS (Toronto Emotional Speech Set) yang terdiri dari audio rekaman yang sudah dilabeli dengan kelas emosi tertentu. Setiap rekaman berisi informasi tentang intonasi dan nada suara yang dapat menggambarkan emosi seperti marah, sedih, bahagia, takut, jijik, netral, dan terkejut.
Jumlah data per-kelas emosi sebanyak 400 file dalam format `.WAV` (Waveform audio format) jadi total keseluruhan data sebanyak 2800 file `.WAV`. Jumlah data yang seimbang ini akan sangat berguna pada proses pelatihan sehingga tidak akan menimbulkan overfitting pada kelas-kelas data tertentu.

Data audio ini merupakan rekaman suara dari 2 orang wanita berusia 26 dan 64 tahun dengan mengucapkan sebuah kalimat `say the word ...` dengan intonasi dan nada yang bervariasi sehingga mewakili ketujuh emosi tersebut.  
Dataset ini dapat diunduh dari sumber berikut: [Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

### Variabel-Variabel dalam Dataset

Dataset audio biasanya tidak memiliki variabel seperti pada data tabular, tetapi terdapat label yang menunjukkan emosi yang direpresentasikan dalam rekaman. Setiap rekaman kemudian diekstrak fiturnya, sehingga variabel fitur yang digunakan adalah:

- **MFCC**: Mel-Frequency Cepstral Coefficients, yang mengukur amplitudo suara pada berbagai frekuensi.
- **Spektral Bandwidth**: Menggambarkan lebar rentang frekuensi dari sinyal.
- **Spektral Rolloff**: Frekuensi di bawah sejumlah persentase dari energi total spektrum.
- **Spektral Contrast**: Perbedaan frekuensi tertinggi dan terendah dari spektrum audio.
- **Spektral Centroid**: Merupakan pusat massa frekuensi dalam spektrum sinyal audio.
- **Zero Crossing Rate**: Merupakan jumlah sinyal yang melewati titik 0 (nol) yang megindikasikan kelembutan suara.

## Data Preparation

Tahapan data preparation yang dilakukan meliputi:
1. **Pengaturan Direktori dan DataFrame (file path dan label)** : Sebelum melakukan pemrosesan data, perlu diatur pengelompokkan file ke dalam direktori yang sesuai dengan jenis labelnya. kemudian membuat DataFrame awal berisi lokasi file beserta labelnya.
2. **Ekstraksi Fitur**: Setiap file audio diekstraksi menggunakan library [librosa](https://librosa.org/doc/latest/index.html) seperti MFCC, spektral bandwidth, spektral rolloff, spektral centroid, dan zero crossing rate (zcr).
3. **Normalisasi Data**: Semua fitur dinormalisasi untuk meningkatkan performa model. Teknik normalisasi ini dilakukan agar rentang fitur tidak terlalu bervariasi, yang dapat mempersulit model dalam menemukan pola.
4. **Split Dataset**: Dataset dibagi menjadi data latih dan validasi untuk melatih dan menguji performa model.

## Modeling

Pada proyek ini, dua model diterapkan untuk klasifikasi emosi:

### Random Forest

Random Forest dipilih karena algoritma ini terkenal dalam klasifikasi data dengan noise.

Kelebihan dari Random Forest:
- Mudah untuk diimplementasikan dan sangat cocok untuk data dengan dimensi yang lebih sedikit.
- Kurang rentan terhadap overfitting pada data.

### Dense Neural Network (DNN)

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

Pada pelatihan menggunakan Neural Network Loss function yang digunakan adalah `Sparse Categorical Crossentropy`, 
formula :

![image](https://github.com/user-attachments/assets/c2f372ad-0132-4536-baa2-5ccc7df2e2e3)

Keterangan:
![image](https://github.com/user-attachments/assets/afba54d7-9e3b-48b5-8d08-0181acfe2ed9)


Penulisan ini akan tampil dengan format matematika yang rapi di Markdown.


Proyek ini menggunakan beberapa metrik evaluasi untuk menilai performa model:

1. **Akurasi**: Proporsi prediksi benar terhadap semua prediksi.
2. **Precision**: Proporsi prediksi benar dari total prediksi positif.
3. **Recall**: Kemampuan model dalam menemukan semua sampel positif.
4. **F1-score**: Harmonik rata-rata precision dan recall, terutama berguna untuk mengatasi ketidakseimbangan data.

### Hasil Evaluasi

| Model          | Precision | Recall | F1-Score | Accuracy |
|----------------|-----------|--------|----------|----------|
| Random Forest  | 0.96      | 0.96   | 0.96     | 96%      |
| Neural Network | 0.97      | 0.97   | 0.97     | 97%      |

Secara keseluruhan hasil performa model dijelaskan sebagai berikut :
**Akurasi**
`Random Forest`: Akurasi 0.96 atau 96%.
`Neural Network`: Akurasi 0.97 atau 97%.
Maka dari itu Neural Network memiliki akurasi sedikit lebih tinggi dibandingkan dengan Random Forest, menunjukkan bahwa model Neural Network lebih mampu menangkap pola dalam data ini.

**Recall, F1-Score, Precision**
`Random Forest` menunjukkan performa baik namun sedikit lebih bervariasi di beberapa kelas (misalnya, pada kelas "happy" dan "surprise").
`Neural Network` memiliki performa lebih konsisten di berbagai kelas, dengan hampir semua kelas memiliki precision, recall, dan F1-score di atas 0.94, serta lebih tinggi pada kelas yang lebih sulit seperti "surprise."

## Kesimpulan

Proyek ini berhasil mencapai tujuan utama, yaitu membangun model yang mampu mengklasifikasikan emosi berdasarkan data audio dengan akurasi yang memadai. Random Forest dan Neural Network masing-masing memiliki kelebihan dalam klasifikasi ini, dan dengan melakukan optimasi hyperparameter, performa model dapat ditingkatkan lebih lanjut.
