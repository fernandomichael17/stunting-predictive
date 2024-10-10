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
### Problem Statements
1. Pernyataan Masalah 1 : Bagaimana mendeteksi gejala stunting pada balita menggunakan model machine learning dari fitur-fitur yang ada?
2. Pernyataan Masalah 2 : Bagaimana cara mengetahui bahwa model yang dibuat dapat benar-benar dikatakan baik dalam memprediksi gejala stunting pada anak sehingga dapat diterapkan ? 
***
### Goals
1. Membangun model machine learning yang mampu mengenali gejala stunting berdasarkan faktor-faktor seperti tinggi badan, Jenis Kelamin, serta Usia (Umur), dan juga Status Gizi sebagai variabel target untuk diprediksi.
2. Membuat sebuah evaluasi dengan beberapa parameter sehingga hasil yang diperoleh dari model dapat dipercaya untuk diterapkan.
***
### Solution Statements
1. Menggunakan algoritma machine learning seperti Random Forest, XGBoost, Decision Tree untuk membandingkan performa model dalam mendeteksi gejala stunting.
2. Menggunakan metrik evaluasi seperti accuracy, precision, dan recall untuk mengukur kinerja setiap model dalam mendeteksi gejala stunting.
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
Dataset ini berisi data mengenai balita, dengan beberapa fitur-fitur diantaranya : 
* Umur :Merupakan umur anak dalam hitungan bulan
* Jenis Kelamin : Jenis Kelamin anak
* Tinggi badan (cm) : Tinggi badan anak dalam centimeter
* Status Gizi : Status gizi yang dikategorikan menjadi 4 status - 'severely stunting', 'stunting', 'normal', dan 'tinggi'. 'Severely stunting' menunjukkan kondisi sangat serius (<-3 SD), 'stunting' menunjukkan kondisi stunting (-3 SD sd <-2 SD), 'normal' mengindikasikan status gizi yang sehat (-2 SD sd +3 SD), dan 'tinggi' (height) menunjukkan pertumbuhan di atas rata-rata (>+3 SD).

### Mengecek nilai null
Untuk lebih mengenal data disini hal pertama yang di lakukan adalah mengecek Nilai Null atau kosong namun pada data ini tidak terdapat data kosong atau null

### Mengecek Outlier
Selanjutnya adalah mengecek adanya outlier dalam data. Outlier sendiri didapat dengan menghitung batas bawah dan batas atas dari data yang dimiliki. Dengan rumus : <br>
Q1 : merupakan nilai quantile pertama atau nilai yang terletak pada posisi ke 25% dalam data<br>
Q2 : merupakan nilai quantile ketiga atau nilai yang terletak pada posisi ke 75% dalam data<br>
IQR : rentang antara Q1 sampai dengan Q3<br>
Batas bawah : Q1 - (1.5 x IQR)<br>
Batas atas : Q3 + (1.5 x IQR)<br>
Untuk melakukan pengecekan agar lebih mudah disini dibuatlah sebuah visualisasi box plot.<br>
![box-plot](https://private-user-images.githubusercontent.com/113835044/375197968-a9d49123-3f26-4e51-8988-6ebf3ef2de0e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjg1Mjk3NjQsIm5iZiI6MTcyODUyOTQ2NCwicGF0aCI6Ii8xMTM4MzUwNDQvMzc1MTk3OTY4LWE5ZDQ5MTIzLTNmMjYtNGU1MS04OTg4LTZlYmYzZWYyZGUwZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAxMFQwMzA0MjRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zNTRkNzM1NGE4ZDZkOTc4MTI3OTJjYWQwNzFjMTljN2NhNzVhMjAwZjUwMjY2YzEyMTU0MzU0MzRhODMwYzFiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.wthiX2v2pCil4VODDfnqJSEyn-DpTIpgYUUAuUl3swA)

<br>
dari hasil boxplot diatas terlihat bahwa adanya oulier pada kolom tinggi badan, sehingga harus dihapus.