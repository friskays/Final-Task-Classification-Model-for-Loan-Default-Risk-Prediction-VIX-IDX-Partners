# Final-Task-Classification-Model-for-Loan-Default-Risk-Prediction-VIX-IDX-Partners

Project ini merupakan Final Task Virtual Internship Program - Data Scientist di ID/X Partners.
Final Task ini bertujuan membangun model yang dapat memprediksi credit risk menggunakan dataset yang terdiri dari data pinjaman yang diterima dan yang ditolak.

# Problem Statement

- Masalah bisnis yang sedang dihadapi adalah terkait pinjaman yang diberikan kepada customer, sebagian diantaranya mengalami keterlambatan/ kesulitan pembayaran. 

Goals : 
- Memperkecil resiko kredit akibat terjadinya gagal pembayaran credit dari customer

Objective :
- Melakukan prediktif model untuk mengetahui customer yang mengalami kesulitan bayar dan lancar dalam pembayaran.
- Menemukan faktor penting dari customer yang lancar dalam pembayaran


# Top Insight 
![image](https://user-images.githubusercontent.com/114457985/215334188-8728b914-f4f1-47e3-80ea-e44a22f74fba.png)
![image](https://user-images.githubusercontent.com/114457985/215334147-16b282b6-99f1-4609-a48c-159794dabf41.png)
![image](https://user-images.githubusercontent.com/114457985/215334193-75d9be4c-bb01-4b26-ae32-a36164df53d8.png)

- Semakin banyak total principal received/ total pokok hutang yang telah dibayarkan customer, proporsi keberhasilan pembayarannya juga cenderung tinggi dibandingkan total pokok hutang yang nominalnya lebih kecil. 
- Semakin besar interest rate atau bunga pinjaman, nasabah cenderung mengalami kesulitan bayar. Terlihat dari proporsi keberhasilan pembayaran, dimana bunga pinjaman lebih dari 22% proporsi nasabah mengalami kesulitan bayar lebih dari 20%.
- Nasabah yang melakukan pengajuan pinjaman dana dengan tujuan produktif, proporsi mengalami kesulitan bayar cenderung lebih tinggi dibandingkan dengan tujuan multiguna (kebutuhan konsumtif). Oleh karena itu untuk dapat dianalisis jenis bisnisnya, agar diketahui jenis bisnis yang dimiliki nasabah yang berpotensi mengalami kesulitan pembayaran hutang-pihutang.


# Modeling
- Model yang digunakan pada case ini adalah Light Gradient Boosting Machine atau LGBM
- Metric yang akan digunakan pada model ini adalah AUC_ROC. Selain itu performa model akan dilihat menggunakan nilai dari Gini dan hasil test statistik dengan Kolmogorov Smirnov
- Parameter yang digunakan pada model merupakan hasil dari hypertuning parameter, karena performa model lebih baik setelah dilakukan hypertuning parameter
- Threshold optimal yang digunakan pada case ini adalah 0.422, dimana hasil tersebut didapatkan dengan menggunakan teknik Gmean
- Final Result yang didapatkan dengan mengimplemantasikan pada Data Test didapatkan hasil 
AUC = 0.940 atau 94%, Gini = 0.880 atau 88.8% dan KS=0.884 atau 88.4%

![image](https://user-images.githubusercontent.com/114457985/215334284-0bb1e2e4-8d1b-4a58-a060-8e9e3d899f00.png)

Feature Importance
![image](https://user-images.githubusercontent.com/114457985/215334318-faefdfad-9e33-41b8-84d6-820d715e6a7d.png)
