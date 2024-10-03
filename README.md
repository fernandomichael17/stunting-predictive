# Machine Learning Terapan | Stunting Toddler (Balita) Detection
###### Created by : Fernando Michael Hebert Siregar

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
Dataset yang digunakan disini berasal dari website Kaggle dengan nama dataset [Stunting Toddler (Balita) Detection](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows) dengan penjabaran sebagai berikut : 
| Bagian | Keterangan |
| ------ | ------ |
| Judul | [Stunting Toddler (Balita) Detection](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows)  |
| Lisensi | [MIT](plugins/github/README.md) |
| Jenis Data | CSV |
| Ukuran | 3.14 MB |
| Jumlah baris dan kolom | 120999 Baris x 4 Kolom |
