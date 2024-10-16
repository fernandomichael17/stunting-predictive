# Laporan Proyek Machine Learning - Fernando Michael Hebert Siregar

***
## Domain Proyek
Tema yang diangkat dari proyek ini adalah kesehatan terutama pada balita, dimana masih banyaknya balita di Indonesia yang mengalami gejala stunting oleh karena itu proyek ini dibuat sebagai langkah awal pengenalan gejala stunting bagi para orang tua dan petugas kesehatan.
***
### Latar Belakang
Stunting adalah gangguan pertumbuhan dan perkembangan anak akibat kekurangan gizi dalam jangka panjang. Stunting bisa disebabkan oleh malnutrisi yang dialami ibu saat hamil, atau anak pada masa pertumbuhannya. Menteri Kesehatan Budi Gunadi Sadikin menyoroti lambatnya penurunan angka stunting di Indonesia. Berdasarkan data Kementerian Kesehatan, angka stunting di Indonesia pada tahun 2023 tercatat sebesar 21,5 persen, hanya turun 0,1 persen dari tahun sebelumnya yang sebesar 21,6 persen. Menkes Budi mengungkapkan bahwa salah satu penyebab rendahnya penurunan stunting adalah belum ditemukannya model implementasi yang efektif untuk program yang telah ditetapkan. Ia menilai ada masalah dalam eksekusi di lapangan sehingga program pencegahan stunting tidak berjalan dengan optimal.<br>

Di era pesatnya kemajuan teknologi kecerdasan buatan (AI), masih banyak yang belum menyadari bahwa kecerdasan buatan memiliki beberapa cabang, dan salah satunya adalah machine learning atau pembelajaran mesin. Machine learning, sebagai bagian dari AI, menjadi daya tarik utama karena memiliki kemampuan belajar seperti manusia. Kemampuan ML untuk memperoleh data dan mempelajari informasi yang ada memungkinkannya untuk menjalankan berbagai tugas yang bervariasi, tergantung pada konteks pembelajaran yang telah dilakukan. <br>

Dengan memanfaatkan machine learning dalam memprediksi gejala stunting pada balita diharapkan dapat membantu para orang tua maupun pekerja medis agar lebih mudah dalam mengidentifikasi masalah tersebut.
<br>
<br>
Sumber : 
- [Alodokter](https://www.alodokter.com/stunting)
- [Dinas Kesehatan](https://dinkes.papua.go.id/menkes-budi-soroti-lambatnya-penurunan-angka-stunting-di-indonesia/#:~:text=Berdasarkan%20data%20Kementerian%20Kesehatan%2C%20angka,yang%20sebesar%2021%2C6%20persen.)
- [Cloud Computing Indonesia](https://www.cloudcomputing.id/pengetahuan-dasar/apa-itu-machine-learning)
***
## Business Understanding
***
Tujuan dari pengembangan model prediksi ini adalah untuk membantu orang tua dan petugas kesehatan dalam mendeteksi gejala stunting pada balita secara cepat dan akurat. Dengan mengenali tanda-tanda awal, tindakan preventif dan intervensi medis dapat segera diambil guna meningkatkan kualitas kesehatan anak-anak. Dibutuhkan sebuah solusi yang dapat mengidentifikasi balita yang berpotensi mengalami stunting berdasarkan data seperti pertumbuhan fisik (berat badan, tinggi badan), usia, dan faktor-faktor lainnya.
### Problem Statements
1. Bagaimana mendeteksi gejala stunting pada balita menggunakan model machine learning dari fitur-fitur yang ada?
2. Dari model yang dilatih model mana yang memiliki akurasi paling baik ?
3. Bagaimana cara mengetahui bahwa model yang dibuat dapat benar-benar dikatakan baik dalam memprediksi gejala stunting pada anak sehingga dapat diterapkan ? 
***
### Goals
1. Membangun model machine learning yang mampu mengenali gejala stunting berdasarkan faktor-faktor seperti tinggi badan, Jenis Kelamin, serta Usia (Umur), dan juga Status Gizi sebagai variabel target untuk diprediksi.
2. Membuat sebuah evaluasi dengan beberapa parameter sehingga hasil yang diperoleh dari model dapat dipercaya untuk diterapkan.
***
### Solution Statements
1. Membandingkan Beberapa Algoritma Machine Learning <br>
Untuk mendeteksi gejala stunting pada balita, akan digunakan beberapa algoritma machine learning, yaitu:
* Random Forest, yang mampu menangani dataset dengan banyak fitur dan menghasilkan prediksi yang robust.
* XGBoost, yang dikenal sebagai salah satu algoritma boosting yang sangat efisien dalam menangani data dengan ketidakseimbangan kelas.
* Decision Tree, sebagai algoritma dasar yang mudah diinterpretasikan dan digunakan sebagai benchmark.
2. Evaluasi Model dengan Metrik yang Relevan <br>
Setiap model akan diukur menggunakan metrik evaluasi yang dapat menggambarkan performa model dalam menangani kasus deteksi gejala stunting:
* Accuracy: Mengukur persentase prediksi benar dari seluruh prediksi.
* Precision: Mengukur akurasi dari prediksi positif yang diberikan oleh model.
* Recall: Mengukur seberapa baik model dapat mendeteksi kasus stunting (true positive rate).
***
## Data Understanding
Dataset yang digunakan disini berasal dari website Kaggle dengan nama dataset [Stunting Toddler (Balita) Detection](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows). Dataset ini merupakan kumpulan data berdasarkan rumus z-score penentuan stunting menurut WHO (World Health Organization), yang berfokus pada deteksi stunting pada balita (bayi dibawah lima tahun). Dataset ini terdiri dari 121.000 baris data, yang merinci informasi mengenai umur, jenis kelamin, tinggi badan, dan status gizi balita. Dataset ini bertujuan untuk membantu peneliti, ahli gizi, dan pembuat kebijakan dalam memahami dan mengatasi masalah stunting pada anak-anak di bawah lima tahun.dengan penjabaran sebagai berikut : 
| Bagian | Keterangan | 
| ------ | ------ |
| Judul | [Stunting Toddler (Balita) Detection](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows)  |
| Lisensi | [MIT](plugins/github/README.md) |
| Jenis Data | CSV |
| Ukuran | 3.14 MB |
| Jumlah baris dan kolom | 120999 Baris x 4 Kolom |
<br>
Dataset yang didapatkan ini berisi data mengenai balita dengan beberap fitur-fitur yang dijabarkan diantaranya : 

| No | Kolom | Jumlah | Tipe Data |
| ------ | ------ | ------ | ------ |
| 1 | Umur (bulan) | 120999 | int64 |
| 2 | Jenis Kelamin | 120999 | object |
| 3 | Tinggi Badan (cm) | 120999 | float64 |
| 4 | Status Gizi | 120999 | object | 

* Umur : Merupakan umur anak dalam hitungan bulan
* Jenis Kelamin : Jenis Kelamin anak
* Tinggi badan (cm) : Tinggi badan anak dalam centimeter
* Status Gizi : Status gizi yang dikategorikan menjadi 4 status - 'severely stunting', 'stunting', 'normal', dan 'tinggi'. 'Severely stunting' menunjukkan kondisi sangat serius (<-3 SD), 'stunting' menunjukkan kondisi stunting (-3 SD sd <-2 SD), 'normal' mengindikasikan status gizi yang sehat (-2 SD sd +3 SD), dan 'tinggi' (height) menunjukkan pertumbuhan di atas rata-rata (>+3 SD).

### Mengecek nilai null
Untuk lebih mengenal data disini hal pertama yang di lakukan adalah mengecek Nilai Null atau kosong namun pada data ini tidak terdapat data kosong atau null

### Mengecek Outlier
Selanjutnya adalah mengecek adanya outlier dalam data. Outlier sendiri didapat dengan menghitung batas bawah dan batas atas dari data yang dimiliki. Yang pertama adalah mencari QI, Q3, dan IQR dari masing-masing kolom pada data. Untuk mencari IQR sendiri merupakan rentang antara Q1 sampai dengan Q3 dengan rumus :


$$IQR = Q_3 - Q_1$$

* Q1 : merupakan nilai quantile pertama atau nilai yang terletak pada posisi ke 25% dalam data
* Q3 : merupakan nilai quantile ketiga atau nilai yang terletak pada posisi ke 75% dalam data
<br>

Batas bawah dan batas atas sendiri didapatkan dengan : <br>
Batas bawah : Q1 - (1.5 x IQR)<br>
Batas atas : Q3 + (1.5 x IQR) <br>



Namun disini untuk mempermudah melakukan pengecekan outlier terhadap masing-masing kolom maka dibuatlah sebuah visualisasi box plot seperti pada gambar dibawah ini.<br>
<br>
![box-plot](https://private-user-images.githubusercontent.com/113835044/375197968-a9d49123-3f26-4e51-8988-6ebf3ef2de0e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjg1Mjk3NjQsIm5iZiI6MTcyODUyOTQ2NCwicGF0aCI6Ii8xMTM4MzUwNDQvMzc1MTk3OTY4LWE5ZDQ5MTIzLTNmMjYtNGU1MS04OTg4LTZlYmYzZWYyZGUwZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAxMFQwMzA0MjRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zNTRkNzM1NGE4ZDZkOTc4MTI3OTJjYWQwNzFjMTljN2NhNzVhMjAwZjUwMjY2YzEyMTU0MzU0MzRhODMwYzFiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.wthiX2v2pCil4VODDfnqJSEyn-DpTIpgYUUAuUl3swA)

<br>
dari hasil boxplot diatas terlihat bahwa adanya outlier pada kolom tinggi badan, sehingga harus dihapus.

### Univariate Analysis
![eda-1](https://private-user-images.githubusercontent.com/113835044/375198591-c72ce087-4bcc-4734-a1ef-b9b0217b0bed.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjg2MTY4MDMsIm5iZiI6MTcyODYxNjUwMywicGF0aCI6Ii8xMTM4MzUwNDQvMzc1MTk4NTkxLWM3MmNlMDg3LTRiY2MtNDczNC1hMWVmLWI5YjAyMTdiMGJlZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAxMVQwMzE1MDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jZWVjNTk4MGM3MGU4ZWNmYWE0YjI2ZDQ1NjgzZmQ3NWFkZjVjNDY1MjgwNWQ4ZmZjZmQzZjYyYThjMjExZTQzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.p5Mj-h2q9IoeYg-BUfs3nX6WssSvDORc34n1hfazRns)
<br>
Dari hasil visualisasi grafik diatas dapat diketahui bahwa dsitribusi data pada jenis kelamin perempuan dan laki-laki normal seimbang, dengan kurang lebih masing-masing pada perempuan berjumlah . 
***
![eda-2](https://private-user-images.githubusercontent.com/113835044/375198570-cad555b0-72d5-4305-8a51-bcbc538bfb73.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjg4MjQxOTQsIm5iZiI6MTcyODgyMzg5NCwicGF0aCI6Ii8xMTM4MzUwNDQvMzc1MTk4NTcwLWNhZDU1NWIwLTcyZDUtNDMwNS04YTUxLWJjYmM1MzhiZmI3My5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAxM1QxMjUxMzRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xNzExZTVmZmZmNDllZWIyOTNjYTE1MzhjMTQwNDlhZjNiZmU1M2Y1MzUxY2Q4ZGZlN2I0MGQwYjNiMGUyMDM3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.yH657Zj_H0P11XFiL9dpCzh2VnnC56M0iAHCrvus0UE)
<br>
Dari visualisasi diatas menunjukkan bahwa mayoritas balita berada pada kategori gizi normal, tetapi terdapat sebagian balita yang mengalami masalah stunting (baik ringan maupun berat), serta hanya sedikit yang memiliki status gizi sangat baik. Analisis ini penting untuk memahami prevalensi masalah stunting pada populasi balita.
***
![eda-3](https://private-user-images.githubusercontent.com/113835044/375198550-440c3005-4948-4838-abf5-a01ed8e32601.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjg4MjQxOTQsIm5iZiI6MTcyODgyMzg5NCwicGF0aCI6Ii8xMTM4MzUwNDQvMzc1MTk4NTUwLTQ0MGMzMDA1LTQ5NDgtNDgzOC1hYmY1LWEwMWVkOGUzMjYwMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAxM1QxMjUxMzRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02NDYwOGRiZDY5YTc4ZWM4OTNiZjkxZjE3OGVjYzBkOGZjNzE4MjNkMGNmYjE5OGUxN2VhODYzYTkzYWY1ZTE4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.aOSXjcSf5kcfOrhVABFFNyRAo1JJxj7-1B-qjpMvuWs)
<br>
Data umur tampaknya memiliki distribusi yang seragam di setiap interval, kecuali pada beberapa bagian tertentu yang lebih rendah. Selanjutnya Terdapat gap yang besar pada bagian tertentu data umur, mungkin terjadi karena kesalahan pencatatan atau kelompok umur tertentu tidak terwakili. Selain itu pada data tinggi badan menunjukkan bentuk yang mendekati normal atau distribusi dengan puncak yang berada di sekitar tinggi badan tertentu. Dapat dilihat bahwa sebagian besar individu dalam dataset memiliki tinggi badan sekitar 80-100 cm, dan jumlahnya menurun pada tinggi badan yang lebih rendah atau lebih tinggi dari itu.

### Multivariate Analysis
