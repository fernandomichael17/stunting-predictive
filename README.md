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

![boxplot](https://github.com/user-attachments/assets/35b55873-9a68-4161-883d-ec52768142b0)

Outlier diidentifikasi dengan metode IQR, menggunakan rumus:

```
IQR = Q3 - Q1
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
```

Outlier dapat memengaruhi performa model dan interpretasi data. Kolom `Tinggi Badan` dibersihkan dari nilai ekstrim ini.

#### Univariate Analysis

- Jenis Kelamin: Distribusi cukup seimbang, dengan 'perempuan' sebanyak 60964 (50.4%) dan 'laki-laki' sebanyak 59997 (49.6%).
  ![univariate](https://github.com/user-attachments/assets/27d15413-7d60-43bc-aa51-99b06b0c7968)

- Status Gizi (Target): Distribusi kelas tidak seimbang. Sebelum sampling, kelas `normal` mendominasi (56.0%), diikuti oleh `severely stunted` (16.4%), `tinggi` (16.2%), dan `stunted` (11.4%). Ketidakseimbangan ini diatasi pada tahap Data Preparation dengan melakukan sampling.<br>
  ![pie](https://github.com/user-attachments/assets/43659aeb-f225-4a58-9e96-8cb1fc40f573)

- Umur (bulan): Distribusi umur balita tampak relatif seragam dengan beberapa penurunan pada interval tertentu.
- Tinggi Badan (cm): Distribusi tinggi badan mendekati bentuk normal, dengan puncak di sekitar 80-100 cm.
  ![hist](https://github.com/user-attachments/assets/f04e1c59-03ee-4e07-8734-e2a49b80e741)


#### Multivariate Analysis

- Dari countplot antara `Status Gizi` dan `Jenis Kelamin`, terlihat bahwa proporsi status gizi pada laki-laki dan perempuan relatif seimbang. Namun, terdapat perbedaan kecil pada kelas `stunted`. `tinggi` dan `normal`, di mana lebih banyak perempuan dibandingkan laki-laki. Namun seperti pada analisis univariate, kelas `normal` mendominasi. <br>
  ![statusvsgender](https://github.com/user-attachments/assets/639c98bf-eeba-4301-9222-bf41f4dbda90)

- Pairplot menunjukkan adanya korelasi positif antara fitur Umur (bulan) dan Tinggi Badan (cm), yang mengindikasikan bahwa seiring bertambahnya usia balita, tinggi badannya cenderung meningkat. <br>
  ![pairplot](https://github.com/user-attachments/assets/21108c63-84b1-4de3-80fd-424f3722efb4)


---

## Data Preparation

1. Handling Data Outlier:

   - Fitur `Tinggi Badan (cm)` dibersihkan dari outlier menggunakan metode IQR (Interquartile Range).
   - Outlier dihapus untuk memastikan model tidak terpengaruh oleh nilai ekstrim yang dapat mengganggu proses pelatihan.
     ![after_boxplot](https://github.com/user-attachments/assets/29094021-5609-4c97-bfb8-6f8c4658f565)


2. **Encoding Data Kategorikal**:

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

3. **Standarisasi Fitur Numerik**:

   - Fitur numerik (`Umur (bulan)`, `Tinggi Badan (cm)`) distandarisasi.
   - Metode yang digunakan adalah **StandardScaler** dari `sklearn.preprocessing`. StandardScaler mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1.
   - Alasan: Beberapa algoritma machine learning, seperti XGBoost, sensitif terhadap skala fitur. Standarisasi membantu memastikan bahwa semua fitur memberikan kontribusi yang seimbang selama proses pelatihan model.

   ```python
   from sklearn.preprocessing
   df[numerical_col] = scaler.fit_transform(df[numerical_col])
   ```

4. **Sampling Data untuk Keseimbangan Kelas**:

   - Setelah pra-pemrosesan awal, dilakukan sampling untuk menyeimbangkan distribusi kelas pada variabel target `Status Gizi`.
   - Dari setiap kategori `Status Gizi` ('normal', 'severely stunted', 'stunted', 'tinggi'), diambil sampel sebanyak 5.000 data.
   - Dataset akhir setelah sampling memiliki 20.000 baris.
   - Alasan: Mengatasi ketidakseimbangan kelas untuk mencegah model menjadi bias terhadap kelas mayoritas dan meningkatkan kemampuannya dalam memprediksi kelas minoritas.

5. **Pembagian Data (Split Data)**:

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

![random_forest](https://github.com/user-attachments/assets/d54640a9-9e2c-42ed-bded-7311f9754673)

#### Classification Report

| Kelas             | Precision  | Recall | F1-Score |
| ----------------- | ---------- | ------ | -------- |
| Normal            | 0.99       | 0.97   | 0.98     |
| Stunting          | 1.00       | 0.99   | 1.00     |
| Severely Stunting | 0.99       | 1.00   | 0.99     |
| Tinggi            | 0.98       | 1.00   | 0.99     |
| **Akurasi**       | **0.9895** |        |          |
| **Macro Avg**     | 0.99       | 0.99   | 0.99     |
| **Weighted Avg**  | 0.99       | 0.99   | 0.99     |

### Evaluasi Model XGBoost

Hasil model XGBoost pada data uji menunjukkan akurasi 87.975%. Berikut adalah hasil evaluasi model melalui Confusion Matrix dan Classification Report.

#### Confusion Matrix

![xgboost](https://github.com/user-attachments/assets/2ca4e7df-2d89-4334-9c56-dcc5cd91df13)

#### Classification Report

| Kelas             | Precision   | Recall | F1-Score |
| ----------------- | ----------- | ------ | -------- |
| Normal            | 0.98        | 0.80   | 0.88     |
| Stunting          | 0.79        | 0.94   | 0.86     |
| Severely Stunting | 0.87        | 0.78   | 0.82     |
| Tinggi            | 0.90        | 0.99   | 0.94     |
| **Akurasi**       | **0.87625** |        |          |
| **Macro Avg**     | 0.89        | 0.88   | 0.88     |
| **Weighted Avg**  | 0.88        | 0.88   | 0.88     |

### Evaluasi Model Decision Tree

Hasil model Decision Tree pada data uji menunjukkan akurasi 98.275%. Berikut adalah hasil evaluasi model melalui Confusion Matrix dan Classification Report.

#### Confusion Matrix

![dtree](https://github.com/user-attachments/assets/1a208f8a-8739-4704-a8e1-e6ed41775f84)

#### Classification Report

| Kelas             | Precision  | Recall | F1-Score |
| ----------------- | ---------- | ------ | -------- |
| Normal            | 0.99       | 0.96   | 0.98     |
| Stunting          | 0.99       | 0.99   | 0.99     |
| Severely Stunting | 0.97       | 0.98   | 0.98     |
| Tinggi            | 0.98       | 1.00   | 0.99     |
| **Akurasi**       | **0.9825** |        |          |
| **Macro Avg**     | 0.98       | 0.98   | 0.98     |
| **Weighted Avg**  | 0.98       | 0.98   | 0.98     |

### Perbandingan Akurasi Model

Tabel berikut menunjukkan perbandingan akurasi dari ketiga model pada data latih dan data uji:

| Model         | Akurasi Data Latih | Akurasi Data Uji |
| :------------ | :----------------- | :--------------- |
| Random Forest | 1.0                | 0.9895           |
| XGBoost       | 0.888813           | 0.87625          |
| Decision Tree | 1.0                | 0.9825           |

![evaluasi](https://github.com/user-attachments/assets/3b646498-3da5-41aa-959c-d7c47d61c64e)

Dari perbandingan di atas, model Random Forest memiliki akurasi tertinggi pada data uji, yaitu 98.65%. Hal ini menunjukkan bahwa model ini memiliki kemampuan generalisasi yang baik terhadap data yang tidak terlihat sebelumnya. Meskipun Decision Tree juga menunjukkan akurasi yang tinggi pada data uji, Random Forest lebih stabil dan kurang rentan terhadap overfitting.

## Kesimpulan

Proyek ini berhasil mengembangkan model klasifikasi untuk memprediksi status gizi balita menggunakan data tinggi badan dan umur. Model Random Forest menunjukkan performa terbaik dengan akurasi 98.95% pada data uji, diikuti oleh Decision Tree (98.25%) dan XGBoost (87.625%). Proses pra-pemrosesan yang dilakukan, termasuk penanganan outlier, encoding fitur kategorikal, standarisasi fitur numerik, dan sampling untuk keseimbangan kelas, berkontribusi pada keberhasilan model. Model ini dapat digunakan sebagai alat bantu dalam pemantauan status gizi balita di Indonesia.
