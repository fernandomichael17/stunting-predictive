# 📝 Laporan Proyek Machine Learning - Fernando Michael Hebert Siregar

---

## Domain Proyek

Tema proyek ini adalah **kesehatan anak**, khususnya **stunting** pada balita. Di Indonesia, kasus stunting masih tergolong tinggi dan menjadi salah satu tantangan dalam pembangunan kesehatan masyarakat. Proyek ini bertujuan untuk memanfaatkan teknologi machine learning guna mendeteksi gejala stunting secara dini.

Stunting adalah kondisi gangguan pertumbuhan akibat kekurangan gizi kronis. Berdasarkan data Kementerian Kesehatan (2023), prevalensi stunting sebesar 21,5%, menurun tipis dari tahun sebelumnya (21,6%). Salah satu tantangan dalam penanggulangan stunting adalah belum optimalnya pelaksanaan program di lapangan. Oleh karena itu, diperlukan pendekatan teknologi seperti machine learning untuk mendukung proses identifikasi secara cepat dan akurat.

📚 Referensi:

- [Alodokter](https://www.alodokter.com/stunting)
- [Dinas Kesehatan](https://dinkes.papua.go.id/menkes-budi-soroti-lambatnya-penurunan-angka-stunting-di-indonesia)
- [Cloud Computing Indonesia](https://www.cloudcomputing.id/pengetahuan-dasar/apa-itu-machine-learning)

---

## Business Understanding

### Problem Statements

1. Bagaimana mendeteksi gejala stunting pada balita menggunakan model machine learning dari fitur-fitur yang ada?
2. Model machine learning mana yang memberikan akurasi prediksi terbaik?
3. Bagaimana memastikan model yang digunakan layak diterapkan di dunia nyata?

### Goals

1. Membangun model machine learning untuk mengenali gejala stunting berdasarkan fitur-fitur seperti umur, tinggi badan, dan jenis kelamin.
2. Mengevaluasi performa model menggunakan metrik evaluasi klasifikasi agar hasil prediksi dapat diandalkan.
3. Memilih model terbaik berdasarkan evaluasi dan validasi.

### Solution Statements

- Menggunakan 3 algoritma machine learning: Decision Tree, Random Forest, dan XGBoost.
- Melakukan evaluasi model menggunakan metrik: Accuracy, Precision, Recall, dan F1-Score.
- Memilih model dengan performa terbaik dan analisis hasil klasifikasinya (Confusion Matrix, Classification Report).

---

## Data Understanding

Dataset yang digunakan berasal dari [Kaggle - Stunting Toddler (Balita) Detection](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows). Dataset terdiri dari 120.999 baris dan 4 kolom, dengan fitur:

| Fitur             | Deskripsi                                                               | Tipe Data |
| ----------------- | ----------------------------------------------------------------------- | --------- |
| Umur (bulan)      | Usia balita dalam bulan                                                 | int64     |
| Jenis Kelamin     | Laki-laki / Perempuan                                                   | object    |
| Tinggi Badan (cm) | Tinggi balita                                                           | float64   |
| Status Gizi       | Target klasifikasi: `severely stunting`, `stunting`, `normal`, `tinggi` | object    |

### Pengecekan Data

#### Pengecekan nilai null :

Berdasarkan hasil pengecekan menggunakan `df.isna().sum()`, diketahui bahwa tidak terdapat nilai null pada setiap kolom dalam dataset.

#### Pengecekan Outlier

- Fitur Umur (bulan): Tidak ditemukan outlier.
- Fitur Tinggi Badan (cm): Ditemukan 38 outlier. Outlier ini kemudian dihapus pada tahap Data Preparation.

![Outlier](https://private-user-images.githubusercontent.com/113835044/447212048-a8a367cc-581a-481e-aff4-38305386d5a5.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTcxODcsIm5iZiI6MTc0ODA1Njg4NywicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEyMDQ4LWE4YTM2N2NjLTU4MWEtNDgxZS1hZmY0LTM4MzA1Mzg2ZDVhNS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMzIxMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02ZGJlYjQwMmNiYzI4MjM0YmMyOGZjMDBiZWViYjhmOWRjYjE2YzQ3ZTc0M2EwYzdiOTUzNWY5ZjVkNzVkMTM5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.RAHA2skgicbso6La68pcNH1iEy1ruB2g_iUDHa9sEMQ)
Outlier diidentifikasi dengan metode IQR, menggunakan rumus:

```
IQR = Q3 - Q1
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
```

Outlier dapat memengaruhi performa model dan interpretasi data. Kolom `Tinggi Badan` dibersihkan dari nilai ekstrim ini.

#### Univariate Analysis

- Jenis Kelamin: Distribusi cukup seimbang, dengan 'perempuan' sebanyak 60964 (50.4%) dan 'laki-laki' sebanyak 59997 (49.6%).
  ![Jenis Kelamin](https://private-user-images.githubusercontent.com/113835044/447212047-ae7617f3-1e2a-4bf1-8c21-0414f2e5a78f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTcxODcsIm5iZiI6MTc0ODA1Njg4NywicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEyMDQ3LWFlNzYxN2YzLTFlMmEtNGJmMS04YzIxLTA0MTRmMmU1YTc4Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMzIxMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mMTg5MWYzYWVmNGU3NTMzYzRmNzA0ZjE2NjVjMGZjZDMwMDc0N2MxZWI0N2E2NThmZmE3M2U5ZDliNmVkMjRkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.ML_-6GQpY_aPErUKE-uK7VzgavMifqPhxjPUaKU-s_k)
- Status Gizi (Target): Distribusi kelas tidak seimbang. Sebelum sampling, kelas `normal` mendominasi (56.0%), diikuti oleh `severely stunted` (16.4%), `tinggi` (16.2%), dan `stunted` (11.4%). Ketidakseimbangan ini diatasi pada tahap Data Preparation dengan melakukan sampling.<br>
  ![Status Gizi](https://private-user-images.githubusercontent.com/113835044/447212050-1a575a07-e9f1-4c02-9a96-dbb0d31686a3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTcxODcsIm5iZiI6MTc0ODA1Njg4NywicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEyMDUwLTFhNTc1YTA3LWU5ZjEtNGMwMi05YTk2LWRiYjBkMzE2ODZhMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMzIxMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kMTdhMGE3NjI2NGM1ZGE0N2FjNmUyYzczODMxZGFjNTFmNWEwNjM2Zjk1MTQ1NWYzYTc4MzMzZTBiODU2MGJlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.VunCoq0BYMb0Nuc9ybg8l_60SmutVk1PfZbU9RxfXiQ)
- Umur (bulan): Distribusi umur balita tampak relatif seragam dengan beberapa penurunan pada interval tertentu.
- Tinggi Badan (cm): Distribusi tinggi badan mendekati bentuk normal, dengan puncak di sekitar 80-100 cm.
  ![Tinggi Badan](https://private-user-images.githubusercontent.com/113835044/447212049-86c60b4b-4d08-4064-9369-603bd02d2ec8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTcxODcsIm5iZiI6MTc0ODA1Njg4NywicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEyMDQ5LTg2YzYwYjRiLTRkMDgtNDA2NC05MzY5LTYwM2JkMDJkMmVjOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMzIxMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05NTlkNWE0OWU3MTNkYTFlYjM3MzFiNDhiZGI4N2Q1ZmQ0NjIyNmFjNTkxODAwMTBhNWZiNmNhMDRjOGIzYmNhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.f0WDBZ7uxhy5fQjf8DUOG7n5UtAF1xLalmwooxvpmus)

#### Multivariate Analysis

- Dari countplot antara `Status Gizi` dan `Jenis Kelamin`, terlihat bahwa proporsi status gizi pada laki-laki dan perempuan relatif seimbang. Namun, terdapat perbedaan kecil pada kelas `stunted`. `tinggi` dan `normal`, di mana lebih banyak perempuan dibandingkan laki-laki. Namun seperti pada analisis univariate, kelas `normal` mendominasi.
  ![Status Gizi vs Jenis Kelamin](https://private-user-images.githubusercontent.com/113835044/447212053-c8e962ed-81ce-460b-a1db-f37a0285ae41.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTcxODcsIm5iZiI6MTc0ODA1Njg4NywicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEyMDUzLWM4ZTk2MmVkLTgxY2UtNDYwYi1hMWRiLWYzN2EwMjg1YWU0MS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMzIxMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lN2UxNDNhYmVmYmYyNjQzNTU0Mzg5YjkwZTk2ZjM5MWE4MGVlZWEyYTNkYzdmN2MzNWQzOGJjYWU5NmUwYTkzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.sxCUGs2ZXCH3wX9YtMhsk23sTJg9F1SrXyg4Mo8v9v4)
- Pairplot menunjukkan adanya korelasi positif antara fitur Umur (bulan) dan Tinggi Badan (cm), yang mengindikasikan bahwa seiring bertambahnya usia balita, tinggi badannya cenderung meningkat.
  ![pairplot](https://private-user-images.githubusercontent.com/113835044/447212051-d1575f9c-852d-454b-9df9-1018dc69bda4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTcxODcsIm5iZiI6MTc0ODA1Njg4NywicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEyMDUxLWQxNTc1ZjljLTg1MmQtNDU0Yi05ZGY5LTEwMThkYzY5YmRhNC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMzIxMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kNDUzMDY1ZDFhZjEwMGYwNmU3MmI3OWFmMThiZDg2ZjNlZTlmMDc2N2JiMWNlYzIzYTBiMDU3NGFiZTM0YTgxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.bUEXKbwY4XeXL2-PIruUcKm8T-g9RoZx51XSRrHpWb4)

---

## Data Preparation

1. **Encoding Data Kategorikal**:

   - Fitur kategorikal `Jenis Kelamin` dan `Status Gizi` diubah menjadi representasi numerik.
   - Metode yang digunakan adalah **Label Encoding** dari `sklearn.preprocessing`.
   - Alasan: Algoritma machine learning umumnya memerlukan input berupa data numerik.

   ```python
   # Contoh kode Label Encoding dari notebook
   from sklearn.preprocessing import LabelEncoder
   encoder = LabelEncoder()
   for col in category_col:
       df[col] = encoder.fit_transform(df[col])
   ```

2. **Standarisasi Fitur Numerik**:

   - Fitur numerik (`Umur (bulan)`, `Tinggi Badan (cm)`) distandarisasi.
   - Metode yang digunakan adalah **StandardScaler** dari `sklearn.preprocessing`. StandardScaler mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1.
   - Alasan: Beberapa algoritma machine learning, seperti XGBoost, sensitif terhadap skala fitur. Standarisasi membantu memastikan bahwa semua fitur memberikan kontribusi yang seimbang selama proses pelatihan model.

   ```python
   from sklearn.preprocessing
   df[numerical_col] = scaler.fit_transform(df[numerical_col])
   ```

3. **Sampling Data untuk Keseimbangan Kelas**:

   - Setelah pra-pemrosesan awal, dilakukan sampling untuk menyeimbangkan distribusi kelas pada variabel target `Status Gizi`.
   - Dari setiap kategori `Status Gizi` ('normal', 'severely stunted', 'stunted', 'tinggi'), diambil sampel sebanyak 5.000 data.
   - Dataset akhir setelah sampling memiliki 20.000 baris.
   - Alasan: Mengatasi ketidakseimbangan kelas untuk mencegah model menjadi bias terhadap kelas mayoritas dan meningkatkan kemampuannya dalam memprediksi kelas minoritas.

4. **Pembagian Data (Split Data)**:

   - Dataset yang telah dipersiapkan dibagi menjadi data latih (train) dan data uji (test).
   - Proporsi pembagian adalah 80% untuk data latih (16.000 sampel) dan 20% untuk data uji (4.000 sampel).
   - Parameter `random_state=42` digunakan untuk memastikan reproduktifitas hasil pembagian.
   - Alasan: Untuk melatih model pada satu set data dan mengujinya pada set data yang belum pernah dilihat sebelumnya, sehingga memberikan evaluasi performa model yang lebih objektif.

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

---

## Modeling

Tiga model klasifikasi machine learning digunakan dan dievaluasi dalam proyek ini. Parameter default dari library `scikit-learn` dan `xgboost` digunakan untuk pelatihan awal setiap model.

### 1. Decision Tree

- **Deskripsi**: Decision Tree adalah model non-parametrik yang memprediksi nilai variabel target dengan mempelajari aturan keputusan sederhana yang diambil dari fitur data.
- **Tahapan & Parameter**: Model `DecisionTreeClassifier()` dari `sklearn.tree` dilatih menggunakan data latih (`X_train`, `y_train`) dengan parameter default.
- **Kelebihan**:
  - Mudah diinterpretasikan dan divisualisasikan.
  - Cepat untuk dilatih.
  - Dapat menangani data numerik dan kategorikal.
- **Kekurangan**:
  - Rentan terhadap overfitting, terutama dengan pohon yang dalam.
  - Tidak stabil, perubahan kecil pada data dapat menghasilkan pohon yang berbeda secara signifikan.

### 2. Random Forest

- **Deskripsi**: Random Forest adalah metode ensemble learning yang membangun banyak decision tree selama pelatihan dan mengeluarkan kelas yang merupakan modus dari kelas (klasifikasi) atau prediksi rata-rata (regresi) dari masing-masing pohon.
- **Tahapan & Parameter**: Model `RandomForestClassifier()` dari `sklearn.ensemble` dilatih menggunakan data latih (`X_train`, `y_train`) dengan parameter default.
- **Kelebihan**:
  - Lebih stabil dan akurat dibandingkan Decision Tree tunggal karena mengurangi varians.
  - Efektif dalam menangani dataset besar dan berdimensi tinggi.
  - Kurang rentan terhadap overfitting dibandingkan Decision Tree.
- **Kekurangan**:
  - Lebih sulit diinterpretasikan (black box) dibandingkan Decision Tree tunggal.
  - Membutuhkan lebih banyak waktu dan sumber daya komputasi untuk pelatihan.

### 3. XGBoost (Extreme Gradient Boosting)

- **Deskripsi**: XGBoost adalah implementasi gradient boosting yang dioptimalkan, dikenal karena kecepatan dan performanya yang tinggi. Model yang digunakan dalam notebook adalah `XGBRFClassifier`, yang merupakan implementasi Random Forest menggunakan XGBoost sebagai backend.
- **Tahapan & Parameter**: Model `XGBRFClassifier()` dari `xgboost` dilatih menggunakan data latih (`X_train`, `y_train`) dengan parameter default.
- **Kelebihan**:
  - Umumnya memberikan akurasi yang sangat tinggi (meskipun dalam kasus ini `XGBRFClassifier` tidak menjadi yang teratas).
  - Menangani data hilang secara internal.
  - Memiliki regularisasi untuk mencegah overfitting.
  - Dapat menjalankan pelatihan secara paralel.
- **Kekurangan**:
  - Lebih kompleks dan tuning parameter bisa lebih rumit dibandingkan model yang lebih sederhana.
  - Membutuhkan pemahaman yang baik tentang cara kerjanya untuk optimasi maksimal.

### Pemilihan Model Terbaik

Berdasarkan hasil evaluasi pada data uji (detail di bagian Evaluation):

- **Random Forest**: Akurasi 0.9865
- **XGBoost (`XGBRFClassifier`)**: Akurasi 0.87975
- **Decision Tree**: Akurasi 0.98275

Dari tabel perbandingan akurasi pada data latih dan data uji:

| Model         | Akurasi Data Latih | Akurasi Data Uji |
| :------------ | :----------------- | :--------------- |
| Random Forest | 1.0                | 0.9865           |
| XGBoost       | 0.888813           | 0.87975          |
| Decision Tree | 1.0                | 0.98275          |

Model **Random Forest** menunjukkan akurasi tertinggi pada data uji (98.65%), diikuti oleh Decision Tree (98.275%). Meskipun akurasi latih untuk Random Forest dan Decision Tree adalah 1.0 (yang bisa mengindikasikan overfitting), Random Forest memiliki performa generalisasi yang sedikit lebih baik pada data uji. Model XGBRFClassifier dalam implementasi ini menunjukkan akurasi yang lebih rendah dibandingkan dua model lainnya.

---

## Evaluation

### Metrik Evaluasi yang Digunakan:

- **Accuracy**: Rasio prediksi yang benar.
- **Precision**: Kemampuan model memprediksi kelas positif secara tepat.
- **Recall**: Kemampuan model mendeteksi semua kasus aktual dari suatu kelas.
- **F1-Score**: Rata-rata harmonik dari precision dan recall.

### Formula Evaluasi:

- Accuracy:

  $$
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
  $$

- Precision:

  $$
  Precision = \frac{TP}{TP + FP}
  $$

- Recall:

  $$
  Recall = \frac{TP}{TP + FN}
  $$

- F1 Score:
  $$
  F1 = 2 * \frac{Precision * Recall}{Precision + Recall}
  $$

Dimana:

- TP (True Positive): Jumlah kasus positif yang diprediksi dengan benar.
- TN (True Negative): Jumlah kasus negatif yang diprediksi dengan benar.
- FP (False Positive): Jumlah kasus negatif yang salah diprediksi sebagai positif.
- FN (False Negative): Jumlah kasus positif yang salah diprediksi sebagai negatif.

### Evaluasi Model Random Forest

Hasil model Random Forest pada data uji menunjukkan akurasi 98.65%. Berikut adalah hasil evaluasi model melalui Confusion Matrix dan Classification Report.

#### Confusion Matrix

![Confusion Matrix](https://private-user-images.githubusercontent.com/113835044/447210781-c0bd6fa7-e9b8-4cf8-af44-44a76f064841.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTU4MzYsIm5iZiI6MTc0ODA1NTUzNiwicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEwNzgxLWMwYmQ2ZmE3LWU5YjgtNGNmOC1hZjQ0LTQ0YTc2ZjA2NDg0MS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMjU4NTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iZGE3MjM5ODg5N2RlMTE3NzJmOGZmMGEwY2ViNzgzNTljYzBlMTc3ZTk2NWY2MDA0NzczMTk4YTFmMzY2OTg3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.B4X1jcB3w85tN0D-c_yZFQedo-2xul2sX73TT_oXsvU)

#### Classification Report

| Kelas             | Precision  | Recall | F1-Score |
| ----------------- | ---------- | ------ | -------- |
| Normal            | 1.00       | 0.97   | 0.98     |
| Stunting          | 1.00       | 0.99   | 0.99     |
| Severely Stunting | 0.97       | 0.99   | 0.98     |
| Tinggi            | 0.98       | 1.00   | 0.99     |
| **Akurasi**       | **0.9865** |        |          |
| **Macro Avg**     | 0.99       | 0.99   | 0.99     |
| **Weighted Avg**  | 0.99       | 0.99   | 0.99     |

### Evaluasi Model XGBoost

Hasil model XGBoost pada data uji menunjukkan akurasi 87.975%. Berikut adalah hasil evaluasi model melalui Confusion Matrix dan Classification Report.

#### Confusion Matrix

![Confusion Matrix](https://private-user-images.githubusercontent.com/113835044/447210837-4207e71c-96eb-4352-b1cd-5406a181d012.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTU4MzYsIm5iZiI6MTc0ODA1NTUzNiwicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEwODM3LTQyMDdlNzFjLTk2ZWItNDM1Mi1iMWNkLTU0MDZhMTgxZDAxMi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMjU4NTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xYTZiNWVlNTQ0N2VjODM0MWM5MDgxNDJmMTJhNzQ2ODBlYzQ0YWVkNWRiZGY2YzM2ZDcwYjFhMmU5NWQ3ZWNlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.0KjVuivLkadTn1ealt8leh_11cBqRYn-vwv0PMj0CU4)

#### Classification Report

| Kelas             | Precision   | Recall | F1-Score |
| ----------------- | ----------- | ------ | -------- |
| Normal            | 0.99        | 0.83   | 0.90     |
| Stunting          | 0.81        | 0.90   | 0.85     |
| Severely Stunting | 0.83        | 0.79   | 0.81     |
| Tinggi            | 0.92        | 1.00   | 0.95     |
| **Akurasi**       | **0.87975** |        |          |
| **Macro Avg**     | 0.89        | 0.88   | 0.88     |
| **Weighted Avg**  | 0.88        | 0.88   | 0.88     |

### Evaluasi Model Decision Tree

Hasil model Decision Tree pada data uji menunjukkan akurasi 98.275%. Berikut adalah hasil evaluasi model melalui Confusion Matrix dan Classification Report.

#### Confusion Matrix

![Confusion Matrix](https://private-user-images.githubusercontent.com/113835044/447210838-c0f6ac0e-9dd4-4d53-a8f5-18b0d8ab67f0.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgwNTY3ODksIm5iZiI6MTc0ODA1NjQ4OSwicGF0aCI6Ii8xMTM4MzUwNDQvNDQ3MjEwODM4LWMwZjZhYzBlLTlkZDQtNGQ1My1hOGY1LTE4YjBkOGFiNjdmMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTI0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyNFQwMzE0NDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04MGU0MGU5YzVmZjlmNDA4NTJiOTY2NGYwNDg4ZjJmZDE0NmQyZTNhMjkxZGY5MzJkZTExMjA5MDFlYTVmYWU4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.dh_YDZcxEAXSQR5GmGrgCeB-ktQgtl6PKZW6ppfs2tk)

#### Classification Report

| Kelas             | Precision   | Recall | F1-Score |
| ----------------- | ----------- | ------ | -------- |
| Normal            | 0.98        | 0.97   | 0.98     |
| Stunting          | 0.99        | 0.99   | 0.99     |
| Severely Stunting | 0.97        | 0.99   | 0.98     |
| Tinggi            | 0.98        | 1.00   | 0.99     |
| **Akurasi**       | **0.98275** |        |          |
| **Macro Avg**     | 0.98        | 0.98   | 0.98     |
| **Weighted Avg**  | 0.98        | 0.98   | 0.98     |

### Perbandingan Akurasi Model

Tabel berikut menunjukkan perbandingan akurasi dari ketiga model pada data latih dan data uji:

| Model         | Akurasi Data Latih | Akurasi Data Uji |
| :------------ | :----------------- | :--------------- |
| Random Forest | 1.0                | 0.9865           |
| XGBoost       | 0.888813           | 0.87975          |
| Decision Tree | 1.0                | 0.98275          |

Dari perbandingan di atas, model Random Forest memiliki akurasi tertinggi pada data uji, yaitu 98.65%. Hal ini menunjukkan bahwa model ini memiliki kemampuan generalisasi yang baik terhadap data yang tidak terlihat sebelumnya. Meskipun Decision Tree juga menunjukkan akurasi yang tinggi pada data uji, Random Forest lebih stabil dan kurang rentan terhadap overfitting.

## Kesimpulan

Proyek ini berhasil mengembangkan model klasifikasi untuk memprediksi status gizi balita menggunakan data tinggi badan dan umur. Model Random Forest menunjukkan performa terbaik dengan akurasi 98.65% pada data uji, diikuti oleh Decision Tree (98.275%) dan XGBoost (87.975%). Proses pra-pemrosesan yang dilakukan, termasuk penanganan outlier, encoding fitur kategorikal, standarisasi fitur numerik, dan sampling untuk keseimbangan kelas, berkontribusi pada keberhasilan model. Model ini dapat digunakan sebagai alat bantu dalam pemantauan status gizi balita di Indonesia.
