# -*- coding: utf-8 -*-
"""movie-recommender-dicoding-ml-v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FV0wevIEC4QUd2mpVHnV4za_QH2DuI9Q

# Content Based Recommender System

# Informasi Dataset

Proyek ini menggunakan kumpulan data MovieLens, kumpulan data yang banyak digunakan di bidang sistem pemberi rekomendasi, yang berisi data film dan metadata.

terdapat 2 file :
movies.csv
ratings.csv

Link dataset : [Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system/data)

# Load Dataset

Pada kode berikut untuk memuat dataset melalui platform kaggle notebook.
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""# Read Dataset

Kode berikut untuk memuat dataset (.csv)
"""

data_movies = pd.read_csv("/kaggle/input/movie-recommendation-system/movies.csv")
data_ratings = pd.read_csv("/kaggle/input/movie-recommendation-system/ratings.csv")

"""Kode berikut untuk melihat contoh data pada file movies.csv, dapat dilihat terdapat 3 kolom pada dataframe ini : `movieId`, `title`, `genres`."""

data_movies.head(-10)

"""# Data Preprocessing

Kode berikut untuk melihat ringkasan terkait dataframe movies, dapat dilihat tipe data untuk tiap kolom beserta jumlah baris dataframe movies yaitu sebanyak **62,423**.
"""

data_movies.info()

"""Kode berikut untuk melihat ringkasan terkait dataframe ratings, dapat dilihat tipe data untuk tiap kolom beserta jumlah baris dataframe ratings yaitu sebanyak **25,000,095**."""

data_ratings.info()

"""selanjutnya mengecek apakah dataframe movies ada data yang kosong (null), dapat terlihat bahwa setiap kolom tidak ada data yang kosong."""

data_movies.isnull().sum()

"""sama seperti sebelumnya selanjutnya mengecek apakah dataframe ratings ada data yang kosong (null), dapat terlihat bahwa setiap kolom tidak ada data yang kosong."""

data_ratings.isnull().sum()

"""selanjutnya kita akan mengecek berapa jumlah data movieId yang unik pada kedua dataframe, terdapat selisih antara kedua data tersebut, kita asumsikan bahwa dataframe ratings berisi data film yang telah ditonton oleh user."""

print(f"Jumlah jenis Movies di data Movies  : {data_movies['movieId'].nunique()}")
print(f"Jumlah jenis Movies di data Ratings : {data_ratings['movieId'].nunique()}")

"""# Data Exploration

Kita akan melakukan eksplorasi data untuk mengetahui manakah film yang paling banyak ditonton/diberi rating oleh user. pertama kita akan menghitung data unik yang muncul pada dataframe ratings, kemudian ambil 50 data pertama ke dalam sebuah list. Pada kode berikut untuk mengambil data `movieId`.
"""

arr_mov_most =  data_ratings['movieId'].value_counts()[:50].index
arr_count_most = data_ratings['movieId'].value_counts()[:50].values
arr_mov_most

"""kemudian dari list top 50 data movieId tadi kita gunakan untuk mencari data title ke dalam list."""

arr_title_most = [data_movies.loc[data_movies['movieId']==x]['title'].values[0] for x in arr_mov_most]

"""Tampilkan data hasil eksplorasi tersebut ke dalam bentuk DataFrame, dapat kita lihat bahwa film `Forrest Gump (1994)` menjadi film yang paling banyak ditonton/dirating oleh user disusul oleh `Shawshank Redemption, The (1994)`."""

df = pd.DataFrame({
    "movieId" : arr_mov_most,
    "title" : arr_title_most,
    "jumlah" : arr_count_most
})

df.head(5)

"""# Data Preparation

Selanjutnya kita akan mengubah data pada kolom `genres` menjadi huruf kecil, untuk menjaga konsistensi data.
"""

data_movies['genres'] = data_movies['genres'].apply(lambda x: x.lower())

"""Pada proyek ini akan menggunakan teknik embedding menggunakan FastText kemudian mengunakan Cosien Similariy untuk perhitungan kedekatan antar data, untuk data yang kita embedding adalah data `title` dan `genres`.

Maka dari itu pada kode ini kita akan menggabungkan kedua data pada tiap baris tersebut menjadi satu string, sebelumnya kita hilangkan tanda `|` pada data genres, kemudian akan menyimpan hasil proses tersebut ke kolom `text`.
"""

from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

data_movies["text"] = data_movies["title"] + " " + data_movies["genres"].str.replace("|", " ")
data_movies.head()

"""Selanjutnya kita menerepakan tokenisasi setiap data pada kolom `text` pada dataframe."""

words_token = [str(x).split() for x in data_movies['text']]
words_token[:2]

"""# Training Model Embedding

selanjutnya kita akan melatih model FastText ke data text yang telah kita pecah menjadi token-token, atur agar ukuran vector 30, dengan epochs 10.
"""

model = FastText(words_token, vector_size=30, window=3, min_count=1, epochs=10)

"""Pada kode berikut adalah fungsi untuk mendapatkan data embedding dari hasil training model FastText, jadi nantinya setiap data pada kolom `text` tersebut akan diubah menjadi bentuk vector embedding dengan fungsi ini."""

def get_embedding(text):
    word = str(text).split()
    vector = [model.wv[x] for x in word if x in model.wv]
    if vector:
        return sum(vector) / len(vector)
    else:
        return np.zeros(vector.vector_size)

"""Apply ke setiap kolom `text` untuk mengubah data tersebut menjadi bentuk vector dan simpan ke kolom `embedding`."""

data_movies["embedding"] = data_movies['text'].apply(get_embedding)
data_movies.head()

"""contoh hasil data embedding"""

data_movies["embedding"][0]

"""# Create Similarity Matrix

Selanjutnya data embedding tersebut kita lakukan stack secara vertikal, kemudian membuat `similarity matrix` dari array embedding tersebut.
"""

embedd = np.vstack(data_movies["embedding"].to_numpy())
similarity_matrix = cosine_similarity(embedd)

"""contoh similarity matrix"""

similarity_matrix[:2]

"""agar index pada dataframe berurutan mulai dari index ke-0, menggunakan kode ini."""

data_movies = data_movies.reset_index(drop=True)
data_movies.head(-1)

"""# Testing Data

Terakhir, kita akan menguji coba dengan step:

- Input nama title film
- Mencari movieId yang sesuai dengan title
- Mendapatkan index data movieId tersebut
- Mencari 10 data dari similarity matrix yang paling mirip dengan data input
- Tampilkan data tersebut ke DataFrame

Hasil pengujian menunjukkan metode dengan embedding menggunakan model FastText dan pencarian cosine similarity memiliki hasil yang cukup baik dalam memberikan rekomendasi film yang mirip secara genre.
"""

nama = input("")
movie_id = data_movies[data_movies['title']==nama]['movieId'].values[0]
movie_data = data_movies[data_movies['movieId']==movie_id]["movieId"]
film_index = movie_data.values[0]
similar_movies = similarity_matrix[movie_data.index[0]].argsort()[::-1][1:10]
print(f"Film : {data_movies.loc[data_movies['movieId']==film_index][['movieId', 'title', 'genres']]}")

print("Rekomendasi:")
pd.DataFrame({
    "movieId" : data_movies.iloc[similar_movies]["movieId"],
    "title" : data_movies.iloc[similar_movies]["title"],
    "genres" : data_movies.iloc[similar_movies]["genres"],
}).head(-1)


