# Laporan Proyek *Machine Learning* - Fikri Dean Radityo

## Domain Proyek

Diabetes merupakan salah satu penyakit kronis yang menjadi tantangan utama dalam dunia kesehatan, terutama di Indonesia. Penyakit ini terjadi ketika kadar glukosa darah terlalu tinggi karena tubuh tidak mampu memproduksi atau menggunakan insulin secara efektif [\[2\]](https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes). Data menunjukkan bahwa jumlah penderita diabetes di Indonesia terus meningkat setiap tahunnya. Pada tahun 2017, terdapat 10,3 juta pasien diabetes di Indonesia, dan angka ini diproyeksikan meningkat menjadi 16,7 juta pada tahun 2045 [\[1\]](https://www.siloamhospitals.com/informasi-siloam/artikel/angka-diabetes-di-indonesia-semakin-tinggi-berikut-faktanya-1). Diabetes tipe 2 menjadi penyumbang terbesar kasus ini, dengan komplikasi seperti kerusakan jantung dan ginjal menjadi penyebab utama kematian [\[1\]](https://www.siloamhospitals.com/informasi-siloam/artikel/angka-diabetes-di-indonesia-semakin-tinggi-berikut-faktanya-1). Selain itu, banyak pasien terlambat menyadari kondisi mereka, sehingga penyakit ini sering kali terdeteksi dalam tahap lanjut [\[3\]](https://doi.org/10.1016/j.procs.2022.12.107). Oleh karena itu, teknologi berbasis *Machine Learning* dapat memainkan peran penting dalam mendeteksi diabetes secara lebih dini dan akurat. Dengan menggunakan pendekatan seperti *supervised learning*, sistem berbasis komputer dapat memproses data kesehatan dan memberikan prediksi tentang kemungkinan seseorang menderita diabetes [\[3\]](https://doi.org/10.1016/j.procs.2022.12.107). Teknologi ini memungkinkan dokter untuk mengambil tindakan lebih cepat, sehingga komplikasi serius dapat dicegah [\[3\]](https://doi.org/10.1016/j.procs.2022.12.107).

Masalah ini harus diselesaikan karena lonjakan jumlah penderita diabetes yang signifikan di Indonesia, yang dapat membebani sistem kesehatan dan meningkatkan angka kematian akibat komplikasi jika tidak segera ditangani [\[1\]](https://www.siloamhospitals.com/informasi-siloam/artikel/angka-diabetes-di-indonesia-semakin-tinggi-berikut-faktanya-1). Keterlambatan diagnosis juga menjadi tantangan besar, karena banyak pasien yang baru memeriksakan diri ketika kondisinya sudah parah, yang mengurangi peluang untuk mendapatkan perawatan yang efektif [\[3\]](https://doi.org/10.1016/j.procs.2022.12.107). Oleh karena itu, langkah-langkah seperti peningkatan kesadaran masyarakat tentang pentingnya pemeriksaan kesehatan rutin perlu diambil [\[2\]](https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes). Pemanfaatan teknologi *Machine Learning* untuk deteksi dini diabetes juga sangat diperlukan, dengan menggunakan data kesehatan untuk memberikan prediksi yang lebih akurat [\[3\]](https://doi.org/10.1016/j.procs.2022.12.107). Kolaborasi lintas sektor antara ahli kesehatan, ilmuwan data, dan pemerintah juga penting untuk menciptakan alat yang mudah diakses, terutama di daerah-daerah terpencil [\[3\]](https://doi.org/10.1016/j.procs.2022.12.107). Dengan pendekatan terintegrasi ini, risiko komplikasi diabetes dapat diminimalkan dan kualitas hidup penderita dapat ditingkatkan, sehingga diabetes dapat dihadapi dengan lebih efektif [\[1\]](https://www.siloamhospitals.com/informasi-siloam/artikel/angka-diabetes-di-indonesia-semakin-tinggi-berikut-faktanya-1)[\[3\]](https://doi.org/10.1016/j.procs.2022.12.107).

## Business Understanding

Diabetes merupakan masalah kesehatan yang semakin meningkat di Indonesia, dengan banyak pasien terlambat didiagnosis, yang memperburuk kondisi mereka. Oleh karena itu, dibutuhkan pengembangan model *Machine Learning* untuk mendeteksi diabetes secara dini dan akurat. Model ini akan menganalisis data kesehatan pasien untuk memprediksi risiko diabetes, memungkinkan dokter mengambil tindakan preventif lebih awal.

Manfaat dari model ini adalah membantu pihak medis, rumah sakit, atau pemerintah dalam mendeteksi diabetes lebih cepat, mengurangi komplikasi serius, dan mencegah penyakit berkembang ke tahap parah. Dengan penerapan model ini, diharapkan risiko komplikasi dapat diminimalkan, dan kualitas hidup penderita meningkat.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara memproses *dataset* agar dapat digunakan untuk membangun model *machine learning* untuk klasifikasi penderita diabetes?
- Bagaimana cara memproses *dataset* agar dapat digunakan untuk pembuatan model *machine learning* klasifikasi penderita diabetes?
- Bagaimana cara memperoleh model klasifikasi penderita diabetes dengan performa terbaik?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Melakukan eksplorasi terhadap *dataset* sehingga dapat digunakan dalam pembuatan model *machine learning* klasifikasi penderita diabetes.
- Memproses *dataset* untuk dapat digunakan pada proses training.
- Membangun dan mengevaluasi model klasifikasi yang memiliki performa terbaik dalam memprediksi penderita diabetes.

### Solution statements
- Untuk melakukan eksplorasi terhadap *dataset*, perlu untuk melakukan *exploratory data analysis* yang dimana termasuk melakukan *univariate analysis*, *multivariate analysis*, dan mendapatkan *correlation matrix* untuk fitur-fitur yang ada.
- Untuk memproses *dataset* untuk dapat digunakan pada proses training, perlu untuk melakukan proses *data cleaning* yang dimana termasuk melakukan *removal duplicates and NaN data*, *one-hot encoding*, *handle imbalance data*, *standardization*, dan *train-test split*.
- Untuk membangun dan mengevaluasi model klasifikasi yang memiliki performa terbaik dalam memprediksi penderita diabetes, diperlukan untuk menggunakan beberapa algoritma guna mencari algoritma terbaik yang memiliki performa akurasi yang memumpuni. Pada kasus ini, digunakan tiga algoritma, yaitu `SVM`, `Random Forest`, dan `KNN` yang dimana diharapkan dapat memberikan performa akurasi yang baik.

## Data Understanding
*Dataset* yang digunakan untuk pembangunan model machine learning ini adalah *dataset* "Diabetes prediction *dataset*" yang tersedia di situs web Kaggle pada [tautan](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) ini. *Dataset* tersebut adalah *dataset* kuantitatif yang berisi kolom-kolom yang dapat digunakan untuk memprediksi apakah seseorang menderita diabetes berdasarkan data-data yang diberikan. *Dataset* ini memiliki 10000 baris dan 9 kolom data.

*Dataset* ini cocok untuk membangun model *supervised learning*, khususnya *binary classification*. Dalam kasus ini adalah melakukan klasifikasi penderita diabetes berdasarkan delapan fitur.

Berikut ini adalah informasi lainnya mengenai *dataset* tersebut:

Variabel-variabel pada *Dataset* "Diabetes prediction dataset" adalah sebagai berikut:

*   `Jenis Kelamin (gender)`: Jenis kelamin mengacu pada jenis kelamin biologis individu, yang dapat memengaruhi kerentanan terhadap diabetes. Terdapat tiga kategori: laki-laki, perempuan, dan lainnya.

*   `Umur (age)`: Umur merupakan faktor penting karena diabetes lebih sering terdiagnosis pada orang dewasa yang lebih tua. Rentang usia dalam *dataset* adalah 0-80 tahun.

*   `Hipertensi (hypertension)`: Hipertensi adalah kondisi medis di mana tekanan darah di arteri terus-menerus tinggi. Nilai 0 menunjukkan tidak memiliki hipertensi, sedangkan nilai 1 berarti memiliki hipertensi.

*   `Penyakit Jantung (heart_disease)`: Riwayat merokok dianggap sebagai faktor risiko untuk diabetes dan dapat memperburuk komplikasi yang terkait dengan diabetes. Dalam *dataset*, terdapat lima kategori: tidak saat ini (not current), mantan perokok (former), tidak ada informasi (No Info), perokok saat ini (current), tidak pernah merokok (never), dan pernah merokok (ever).

*   `Riwayat Merokok (smoking_history)`: Penyakit jantung adalah kondisi medis yang juga berhubungan dengan peningkatan risiko diabetes. Nilai 0 menunjukkan tidak memiliki penyakit jantung, sedangkan nilai 1 berarti memiliki penyakit jantung.

*   `BMI (bmi)`: Indeks Massa Tubuh adalah ukuran yang digunakan untuk menilai apakah berat badan seseorang berada dalam kategori sehat berdasarkan tinggi badannya.

*   `Kadar HbA1c (HbA1c_level)`: Kadar HbA1c mencerminkan kadar gula darah rata-rata seseorang dalam beberapa bulan terakhir.

*   `Kadar Glukosa Darah (blood_glucose_level)`: Kadar glukosa darah mengacu pada jumlah gula yang ada dalam darah seseorang pada suatu waktu tertentu.

*   `Diabetes (age)`: Diabetes adalah kondisi kesehatan kronis yang terjadi ketika kadar gula darah seseorang terlalu tinggi. Nilai 0 menunjukkan tidak menderita diabetes, sedangkan nilai 1 berarti menderita diabetes.

Untuk melakukan eksplorasi terhadap dataset, selanjutnya dilakukan *exploratory data analysis* dan visualisasi data.

### *Exploratory Data Analysis*
***Exploratory Data Analysis*** (EDA) adalah tahap awal dalam mengeksplorasi data untuk menganalisis karakteristik, mengidentifikasi pola, menemukan anomali, dan memverifikasi asumsi-asumsi yang ada dalam data.

```python
# Check dataset rows and columns
df.shape
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
(100000, 9)
```

Berdasarkan *output* di atas, didapatkan informasi sebagai berikut:

- Jumlah baris dan kolom adalah 100000 baris dan 9 kolom.
- Terdapat 9 kolom fitur yaitu `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, dan `blood_glucose_level` serta 1 kolom label yaitu `diabetes`.


```python
df.keys()
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
Index(['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
      'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'],
    dtype='object')
```

Berdasarkan *output* di atas, didapatkan informasi sebagai berikut:
- Jumlah baris dan kolom adalah 100000 baris dan 9 kolom.
- Terdapat 9 kolom fitur yaitu gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, dan blood_glucose_level serta 1 kolom label yaitu diabetes.


```python
# Get statistic of the data
df.describe()
```

Kode diatas menghasilkan *output* sebagai berikut:

|           | Age       | Hypertension | Heart Disease | BMI       | HbA1c Level | Blood Glucose Level | Diabetes  |
|-----------|-----------|--------------|---------------|-----------|-------------|---------------------|-----------|
| **Count** | 100000.00 | 100000.00    | 100000.00     | 100000.00 | 100000.00   | 100000.00           | 100000.00 |
| **Mean**  | 41.89     | 0.0749       | 0.0394        | 27.32     | 5.53        | 138.06              | 0.085     |
| **Std**   | 22.52     | 0.2631       | 0.1946        | 6.64      | 1.07        | 40.71               | 0.2789    |
| **Min**   | 0.08      | 0.0000       | 0.0000        | 10.01     | 3.50        | 80.00               | 0.0000    |
| **25%**   | 24.00     | 0.0000       | 0.0000        | 23.63     | 4.80        | 100.00              | 0.0000    |
| **50%**   | 43.00     | 0.0000       | 0.0000        | 27.32     | 5.80        | 140.00              | 0.0000    |
| **75%**   | 60.00     | 0.0000       | 0.0000        | 29.58     | 6.20        | 159.00              | 0.0000    |
| **Max**   | 80.00     | 1.0000       | 1.0000        | 95.69     | 9.00        | 300.00              | 1.0000    |


Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
*   **Count**  adalah jumlah sampel pada data.
*   **Mean** adalah nilai rata-rata.
*   **Std** adalah standar deviasi.
*   **Min** yaitu nilai minimum setiap kolom.
*   **25%** adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
*   **50%** adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
*   **75%** adalah kuartil ketiga.
*   **Max** adalah nilai maksimum.

```python
# Check if there are missing values in the DataFrame
missing_values = df.isnull().sum()
missing_values
```

Kode diatas menghasilkan *output* sebagai berikut:

|   |                  |
|---|------------------|
| gender               | 0
| age                  | 0
| hypertension         | 0
| heart_disease        | 0
| smoking_history      | 0
| bmi                  | 0
| HbA1c_level          | 0
| blood_glucose_level  | 0
| diabetes             | 0

dtype: int64

```python
# Check if there are any missing values in the entire DataFrame
any_missing = df.isnull().values.any()
print(any_missing)
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
false
```

Berdasarkan *output* kode diatas, dapat dilihat bahwa tidak ada baris yang memiliki baris yang berisi value *null* atau *missing value*

### *Data visualization*
#### *Univariate Analysis*
*Univariate analysis* adalah analisis yang hanya melibatkan satu variabel (*feature*) pada satu waktu. Tujuan utamanya adalah memahami distribusi, karakteristik, dan pola variabel tersebut.

Fitur-fitur dibagi menjadi fitur kategorikan dan fitur numerikal untuk mempermudah analisis yang akan dilakukan

```python
# Gather features based on categorical or numerical type
categorical_features = ['gender', 'smoking_history']
numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
```

Fungsi `CoundAndPlot` dibuat guna mengurangi redudansi kode

```python
# Create function to count and plot categorical data

def CountAndPlot(feature):
  count = df[feature].value_counts()
  percent = 100*df[feature].value_counts(normalize=True)
  samples = pd.DataFrame({'Sample Count':count, 'Percentage':percent.round(1)})
  print(samples)
  count.plot(kind='bar', title=feature)

```

```python
# Count and plot gender data
CountAndPlot(categorical_features[0])
```

Kode diatas menghasilkan *output* sebagai berikut:

Kode diatas menghasilkan *output*

| gender | Sample Count | Percentage |
|--------|--------------|------------|
| Female | 58552        | 58.6       |
| Male   | 41430        | 41.4       |
| Other  | 18           | 0.0        |

![Gender Distribution Graph](image.png)

Berdasarkan grafik diatas, dapat dilihat bahwa jumlah data berjenis kelamin **perempuan berjumlah 58552 (58,6 persen)** dan data berjenis kelamin **laki-laki berjumlah 41430 (41,4 persen)**. Selain itu terdapat juga data berjenis kelamin lainnya (*other*)

```python
# Count and plot gender data
CountAndPlot(categorical_features[1])
```

Kode diatas menghasilkan *output* sebagai berikut:

| Smoking History | Sample Count | Percentage |
|-----------------|--------------|------------|
| No Info         | 35816        | 35.8       |
| never           | 35095        | 35.1       |
| former          | 9352         | 9.4        |
| current         | 9286         | 9.3        |
| not current     | 6447         | 6.4        |
| ever            | 4004         | 4.0        |

![Smoking History Distribution Graph](image-1.png)

Berdasarkan grafik diatas, dapat dilihat bahwa jumlah data yang **tidak terdapat informasi mengenai riwayat merokok** dan data yang memiliki riwayat **tidak pernah merokok** mendominasi pada *dataset*

```python
# Observe the relationship between numeric features
df.hist(bins=50, figsize=(20,15))
plt.show()
```

Kode diatas menghasilkan *output* sebagai berikut:

![Numeric Data Distribution](image-2.png)

Fitur-fitur yang termasuk ke dalam data numerikal dapat dilihat pada histogram diatas


#### *Multivariate Analysis*

*Multivariate analysis* adalah analisis yang melibatkan dua atau lebih variabel secara bersamaan. Tujuan utamanya adalah untuk memahami hubungan atau interaksi antara variabel-variabel tersebut.

```python
for col in categorical_features:
  sns.catplot(
      x=col,
      y="diabetes",
      kind="bar",
      dodge=False,
      height=4,
      aspect=3,
      data=df,
      palette="Set3",
      hue=col,
      legend=False
  )
  plt.title("Rata-rata 'diabetes' Relatif terhadap - {}".format(col))
```

Kode diatas menghasilkan *output* sebagai berikut:

![Rata-rata diabetes relatif terhadap gender](image-3.png)

**Rata-rata diabetes relatif terhadap jenis kelamin**

Terlihat bahwa jenis kelamin memiliki pengaruh terhadap rata-rata diabetes. Data berjenis kelamin kaki-laki memiliki rata-rata diabetes yang lebih tinggi dibandingkan data berjenis kelamin perempuan, sementara data berjenis kelamin lainnya (*other*) memiliki rata-rata yang sangat rendah atau hampir nol.

**Rata-rata diabetes relatif terhadap riwayat merokok**

Data yang pernah menjadi perokok (former) memiliki rata-rata diabetes tertinggi dibandingkan kategori lain, sementara data yang tidak memiliki informasi mengenai riwayat merokok memiliki rata-rata diabetes terendah.

Dari kedua grafik diatas, dapat disimpulkan bahwa faktor jenis kelamin dan riwayat merokok dapat menjadi faktor risiko penting yang memengaruhi kemungkinan terkena diabetes.

```python
sns.pairplot(df, diag_kind = 'kde')
```

Kode diatas menghasilkan *output* sebagai berikut:

![dataset pairplot](image-4.png)

```python
sns.pairplot(df, diag_kind = 'kde', hue ='diabetes')
```

Kode diatas menghasilkan *output* sebagai berikut:

![dataset pairplot with diabetes as hue](image-5.png)

Pada gambar kedua (yang menekankan perbedaan warna antara diabetes dan tidak),
Dapat terlihat bahwa variabel seperti HbA1c_level dan blood_glucose_level memiliki hubungan kuat dengan diabetes, di mana nilai yang lebih tinggi pada kedua variabel ini sering dikaitkan dengan kasus diabetes.

```python
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
```

Kode diatas menghasilkan *output* sebagai berikut:

![Correlation matrix](image-6.png)

Pada correlation matrix diatas, dapat dilihat bahwa fitur HbA1c_level dan blood_glucose_level memiliki pengaruh besar untuk prediksi diabetes. Hal ini bisa diambil dengan melihat bahwa terdapat perbedaan signifikan antara penderita diabetes dan tidak pada variabel-variabel tersebut.


## *Data Preperation*
### *Data Cleaning*

*Data Cleaning* adalah proses memperbaiki atau menghapus data yang salah, rusak, memiliki format yang tidak sesuai, duplikat, atau tidak lengkap dalam sebuah *dataset*. Ketika menggabungkan berbagai sumber data, peluang terjadinya duplikasi atau pelabelan yang salah menjadi lebih besar.

**Alasan**: Proses *Data Cleaning* harus dilakukan untuk memeriksa, memperbaiki, dan memastikan bahwa data yang digunakan sudah siap digunakan dan tidak terdapat potensi kesalahan


***Removal Duplicates and NaN Data***
Menghapus entri duplikat memastikan bahwa dataset menjadi unik dan bebas dari baris yang redundan. Duplikasi dapat terjadi akibat kesalahan pengumpulan data, penggabungan dataset, atau duplikasi yang tidak disengaja.

*NaN* (*Not a Number*) adalah nilai yang mewakili data yang hilang atau tidak terdefinisi dalam sebuah dataset. Penanganan data *NaN* sangat penting untuk memastikan bahwa perhitungan dan analisis tidak menjadi bias atau menghasilkan hasil yang tidak akurat.

**Alasan**: Proses ini perlu dilakukan guna membersihkan data yang berpotensi memunculkan kesalahan seperti value *NaN* dan menghilangkan data yang berpotensi menurunkan performa seperti data duplikat.


```python
# Drop duplicate and NaN
clean_df = df.drop_duplicates().dropna()
```

***One-Hot Encoding***

*One-hot encoding* adalah salah satu metode untuk mengubah data sehingga siap digunakan oleh algoritma dan menghasilkan prediksi yang lebih baik. Dalam metode ini, setiap nilai kategori diubah menjadi kolom kategori baru, kemudian diberikan nilai biner 1 atau 0 pada kolom-kolom tersebut.

**Alasan**: Proses ini perlu dilakukan karena algoritma dapat bekerja dengan baik untuk data numerikal sehingga data kategorikal harus diubah menjadi biner seperti 0 dan 1


```python
# One Hot Encoding
for col in categorical_features:
  clean_df = pd.concat([clean_df, pd.get_dummies(clean_df[col], prefix=col, dtype=int)],axis=1)

clean_df.drop(categorical_features, axis=1, inplace=True)
```

Mendapatkan informasi dan Memeriksa *value* dari variable `clean_df` setelah dilakukannya proses *Removal Duplicates and NaN Data* dan *One-Hot Encoding*

```python
clean_df.info()
```

Kode diatas menghasilkan *output* sebagai berikut:

**Index:**  96146 entries, 0 to 99999  

**Data Columns (Total: 16):**

| #   | Column                       | Non-Null Count | Dtype   |
|-----|------------------------------|----------------|---------|
| 0   | age                          | 96146 non-null | float64 |
| 1   | hypertension                 | 96146 non-null | int64   |
| 2   | heart_disease                | 96146 non-null | int64   |
| 3   | bmi                          | 96146 non-null | float64 |
| 4   | HbA1c_level                  | 96146 non-null | float64 |
| 5   | blood_glucose_level          | 96146 non-null | int64   |
| 6   | diabetes                     | 96146 non-null | int64   |
| 7   | gender_Female                | 96146 non-null | int64   |
| 8   | gender_Male                  | 96146 non-null | int64   |
| 9   | gender_Other                 | 96146 non-null | int64   |
| 10  | smoking_history_No Info      | 96146 non-null | int64   |
| 11  | smoking_history_current      | 96146 non-null | int64   |
| 12  | smoking_history_ever         | 96146 non-null | int64   |
| 13  | smoking_history_former       | 96146 non-null | int64   |
| 14  | smoking_history_never        | 96146 non-null | int64   |
| 15  | smoking_history_not current  | 96146 non-null | int64   |

**dtypes:**  float64(3), int64(13)  

**Memory Usage:**  12.5 MB

```python
clean_df.head()
```

Kode diatas menghasilkan *output* sebagai berikut:


| age  | hypertension | heart_disease |   bmi   | HbA1c_level | blood_glucose_level | diabetes | gender_Female | gender_Male | gender_Other | smoking_history_No Info | smoking_history_current | smoking_history_ever | smoking_history_former | smoking_history_never | smoking_history_not current |
|------|--------------|---------------|---------|-------------|---------------------|----------|---------------|-------------|--------------|--------------------------|-------------------------|----------------------|------------------------|-----------------------|----------------------------|
| 80.0 | 0            | 1             | 25.19   | 6.6         | 140                 | 0        | 1             | 0           | 0            | 0                        | 0                       | 0                    | 0                      | 1                     | 0                          |
| 54.0 | 0            | 0             | 27.32   | 6.6         | 80                  | 0        | 1             | 0           | 0            | 1                        | 0                       | 0                    | 0                      | 0                     | 0                          |
| 28.0 | 0            | 0             | 27.32   | 5.7         | 158                 | 0        | 0             | 1           | 0            | 0                        | 0                       | 0                    | 0                      | 1                     | 0                          |
| 36.0 | 0            | 0             | 23.45   | 5.0         | 155                 | 0        | 1             | 0           | 0            | 0                        | 1                       | 0                    | 0                      | 0                     | 0                          |
| 76.0 | 1            | 1             | 20.14   | 4.8         | 155                 | 0        | 0             | 1           | 0            | 0                        | 1                       | 1                    | 0                      | 0                     | 0                          |


***Handle Imbalance Data***
*Imbalanced data* adalah istilah yang digunakan untuk menggambarkan jenis dataset tertentu dan menjadi tantangan utama dalam masalah klasifikasi. *Imbalanced data* mengacu pada situasi di mana jumlah sampel pada setiap kelas sangat bervariasi.

```python
counts = clean_df['diabetes'].value_counts()
labels = ['Diabetes' if val == 1 else 'No Diabetes' for val in counts.index]

plt.figure(figsize=(8, 8))
counts.plot.pie(
    autopct='%1.1f%%',
    labels=labels,
    startangle=90,
    colors=['green', 'red']
)
plt.title('Distribution of Diabetes')
plt.ylabel('')
plt.show()
```

Kode diatas menghasilkan *output* sebagai berikut:

![Distribution of diabetes graph](image-7.png)

Dari *pie chart* diatas dapat dilihat bahwa terdapat imbalance data pada *dataset* ini. Selanjutnya akan dilakukan *oversampling* untuk mengatasi *imbalance data* tersebut.

```python
# Drop the target column
X = clean_df.drop(['diabetes'], axis=1)
y = clean_df['diabetes']
```

```python
# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")
```

Kode diatas menghasilkan *output* sebagai berikut:

Resampled class distribution: diabetes

| diabetes | count  |
|----------|--------|
| 0        | 87664  |
| 1        | 87664  |

Name: count, dtype: int64

***Standardization***

Standarisasi adalah metode penskalaan fitur di mana nilai data diubah skala agar sesuai dengan distribusi antara 0 dan 1, menggunakan nilai rata-rata (*mean*) dan standar deviasi sebagai dasar untuk menghitung nilai tertentu. Jarak antar data kemudian digunakan untuk memetakan kesamaan dan perbedaan.

```python
# Scale the X values using scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
```

```python
X_scaled
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
array([[ 1.3735614 , -0.29101326,  4.94997521, ..., -0.30432046,
         1.54087245, -0.20903013],
       [ 0.16495802, -0.29101326, -0.20202121, ..., -0.30432046,
        -0.64898298, -0.20903013],
       [-1.04364536, -0.29101326, -0.20202121, ..., -0.30432046,
         1.54087245, -0.20903013],
       ...,
       [ 0.72277497, -0.29101326, -0.20202121, ..., -0.30432046,
        -0.64898298, -0.20903013],
       [ 0.62742912,  3.43626949, -0.20202121, ..., -0.30432046,
        -0.64898298, -0.20903013],
       [-0.21626573, -0.29101326,  4.94997521, ..., -0.30432046,
        -0.64898298, -0.20903013]])
```

```python
print(X_resampled.describe())     # Check for extreme values or outliers
print(y_resampled.value_counts()) # Check the class distribution in `y`
```

Kode diatas menghasilkan *output* sebagai berikut:

| Statistic           | age          | hypertension  | heart_disease | bmi         | HbA1c_level  | blood_glucose_level  | gender_Female | gender_Male | gender_Other | smoking_history_No Info  | smoking_history_current| smoking_history_ever | smoking_history_former| smoking_history_never| smoking_history_not current|
|---------------------|--------------|---------------|---------------|-------------|--------------|----------------------|---------------|-------------|--------------|--------------------------|------------------------|----------------------|-----------------------|----------------------|----------------------------|
| count               | 175328       | 175328        | 175328        | 175328      | 175328       | 175328               | 175328        | 175328      | 175328       | 175328                   | 175328                 | 175328               | 175328                | 175328               | 175328                     |
| mean                | 50.451       | 0.078         | 0.039         | 29.432      | 6.146        | 163.494              | 0.484         | 0.371       | 0.000        | 0.214                    | 0.063                  | 0.025                | 0.085                 | 0.296                | 0.042                      |
| std                 | 21.512       | 0.268         | 0.194         | 7.424       | 1.226        | 57.047               | 0.500         | 0.483       | 0.010        | 0.410                    | 0.242                  | 0.156                | 0.279                 | 0.457                | 0.200                      |
| min                 | 0.080        | 0             | 0             | 10.010      | 3.500        | 80                   | 0             | 0           | 0            | 0                        | 0                      | 0                    | 0                     | 0                    | 0                          |
| 25%                 | 36.000       | 0             | 0             | 25.739      | 5.700        | 130                  | 0             | 0           | 0            | 0                        | 0                      | 0                    | 0                     | 0                    | 0                          |
| 50%                 | 54.000       | 0             | 0             | 27.320      | 6.127        | 155                  | 0             | 0           | 0            | 0                        | 0                      | 0                    | 0                     | 0                    | 0                          |
| 75%                 | 67.000       | 0             | 0             | 32.920      | 6.600        | 200                  | 1             | 1           | 0            | 0                        | 0                      | 0                    | 0                     | 1                    | 0                          |
| max                 | 80.000       | 1             | 1             | 95.690      | 9.000        | 300                  | 1             | 1           | 1            | 1                        | 1                      | 1                    | 1                     | 1                    | 1                          |

### *Train Test Split*

*Train-test split* adalah metode sederhana untuk mengukur kinerja algoritma machine learning dalam prediksi dengan membandingkan hasil model yang dibuat dengan data yang belum pernah dilihat sebelumnya.

**Alasan**: Proses ini dilakukan dengan tujuan untuk melakukan evaluasi terhadap model yang akan dibuat dengan mengukur seberapa baik model dapat memprediksi data yang belum pernah dilihat (data yang tidak dimasukkan ke data latih melainkan ke data *test*).

```python
# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2)
```

Data telah dibagi menjadi dua yaitu data latih sebesar 20% dan data *testing* sebesar 80% dari total keseluruhan data.


## Modeling
Setelah proses *Data Preparation* telah dilakukan, selanjutnya adalah proses *Modelling* yang dimana model akan dibuat dan melatih model tersebut dengan menggunakan data *training* serta menggunakan data *testing* untuk melakukan evaluasi terhadap model yang telah dibuat.

Pada proses *modelling* ini, tiga algoritma akan digunakan untuk melatih model. Berikut ketiga algoritma tersebut disertai dengan kelebihan dan kekurangan dari setiap algoritma:

- SVC (Support Vector Classifier)
  - Kelebihan:
    - Baik untuk data berdimensi tinggi.
    - Margin terbesar antara kelas.
    - Kuat terhadap overfitting.
  - Kekurangan:
    - Lambat untuk dataset besar.
    - Memerlukan tuning parameter yang hati-hati.
    - Kurang efisien untuk data besar karena berat di komputasi.

- Random Forest
  - Kelebihan:
    - Tahan terhadap overfitting.
    - Bisa menangani data hilang dan outliers.
    - Mudah menganalisis fitur penting.
  - Kekurangan:
    - Sulit diinterpretasi secara keseluruhan.
    - Waktu pelatihan lama.
    - Memerlukan banyak memori.

- K-Nearest Neighbors (KNN)
  - Kelebihan:
    - Sederhana dan mudah dipahami.
    - Tidak memerlukan pelatihan eksplisit.
    - Fleksibel dengan banyak tipe data.
  - Kekurangan:
    - Waktu inferensi lama.
    - Sensitif terhadap skala data.
    - Rentan terhadap noise dan data tidak seimbang.


Dari ketiga algoritma tersebut, akan dipilih model terbaik yang diukur melalui rata-rata metrik setelah masing-masing model dibuat, dilatih, dan dievaluasi.

**Model *machine learning* menggunakan algoritma `SVC`**

Parameter yang digunakan pada model yang menggunakan algoritma `SVC` adalah parameter default dari algoritma tersebut. Parameter yang digunakan adalah sebagai berikut:

- `C (Regularisasi)`: Parameter regularisasi yang mengontrol keseimbangan antara kesalahan pada data pelatihan dan penyederhanaan model (maksimalisasi margin). Nilai default dari parameter ini adalah `1.0`.
- `kernel`: Jenis fungsi kernel yang digunakan untuk memetakan data ke dimensi yang lebih tinggi. Nilai default dari parameter ini adalah `rbf`.
- `gamma`: Koefisien kernel untuk kernel 'rbf', 'poly', dan 'sigmoid'. Nilai default dari parameter ini adalah `scale`.

```python
svm_model = SVC()

svm_model.fit(X_train, y_train)

y_pred_train_svm = svm_model.predict(X_train)
y_pred_test_svm = svm_model.predict(X_test)

accuracy_train_svm = accuracy_score(y_pred_train_svm, y_train)
accuracy_test_svm = accuracy_score(y_pred_test_svm, y_test)

print('SVC - accuracy_train:', accuracy_train_svm)
print('SVC - accuracy_test:', accuracy_test_svm)

print('Classification Report:\n', classification_report(y_test, y_pred_test_svm))
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
SVC - accuracy_train: 0.946692618100412
SVC - accuracy_test: 0.9472138253578966

Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.97      0.95     17447
           1       0.97      0.93      0.95     17619

    accuracy                           0.95     35066
   macro avg       0.95      0.95      0.95     35066
weighted avg       0.95      0.95      0.95     35066
```

Model SVM dengan konfigurasi **SVC** menunjukkan performa yang sangat baik pada *dataset*.

- Akurasi:
  - Data latih: **94,67%**
  - Data uji: **94,72%**
  - Hasil ini menunjukkan generalisasi yang kuat tanpa indikasi overfitting.

- Data Klasifikasi:
  - **Kelas 0** (Tidak Diabetes):
    - Presisi: **93%**
    - Recall: **97%**
    - F1-Score: **95%**
  - **Kelas 1** (Diabetes):
    - Presisi: **97%**
    - Recall: **93%**
    - F1-Score: **95%**

- Rata-rata metrik untuk seluruh kelas:
  - Presisi: **95%**
  - Recall: **95%**
  - F1-Score: **95%**

- *Dataset* berukuran **35.066 sampel**, dengan keseimbangan performa yang baik antara kedua kelas.

**Model *machine learning* menggunakan algoritma `Random Forest`**

Parameter yang digunakan pada model yang menggunakan algoritma `Random Forest` adalah parameter default dari algoritma tersebut. Parameter yang digunakan adalah sebagai berikut:

- `n_estimators`: Jumlah *tree* pada algoritma `Random Forest`. Semakin besar, semakin baik prediksi, tetapi membutuhkan waktu komputasi lebih lama. Nilai default dari parameter ini adalah `100`.
- `max_depth`: Kedalaman yang ditentukan untuk *tree*. Nilai default dari parameter ini adalah `None`.
- `min_samples_leaf`: Jumlah minimum sampel pada *tree*. Nilai default dari parameter ini adalah `1`.
- `min_samples_split`: Jumlah minimum sampel untuk membagi node internal. Nilai default dari parameter ini adalah `2`.
- `max_features`: Jumlah maksimum fitur yang dipertimbangkan untuk split. Nilai default dari parameter ini adalah `sqrt`.

```python
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

accuracy_train_rf = accuracy_score(y_pred_train_rf, y_train)
accuracy_test_rf = accuracy_score(y_pred_test_rf, y_test)

print('Random Forest - accuracy_train:', accuracy_train_rf)
print('Random Forest - accuracy_test:', accuracy_test_rf)

print('Classification Report:\n', classification_report(y_test, y_pred_test_rf))
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
Random Forest - accuracy_train: 0.9995722291140865
Random Forest - accuracy_test: 0.9792391490332516
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.99      0.98     17483
           1       0.99      0.97      0.98     17583

    accuracy                           0.98     35066
   macro avg       0.98      0.98      0.98     35066
weighted avg       0.98      0.98      0.98     35066
```

Model Random Forest menunjukkan performa yang sangat baik pada *dataset*.

- Akurasi:
  - Data latih: **99,96%**
  - Data uji: **97,92%**
  - Hasil ini menunjukkan bahwa model mampu menggeneralisasi dengan baik pada data uji.

- Data Klasifikasi:
  - **Kelas 0** (Tidak Diabetes):
    - Presisi: **97%**
    - Recall: **99%**
    - F1-Score: **98%**
  - **Kelas 1** (Diabetes):
    - Presisi: **99%**
    - Recall: **97%**
    - F1-Score: **98%**

- Rata-rata metrik untuk seluruh kelas:
  - Presisi: **98%**
  - Recall: **98%**
  - F1-Score: **98%**

- *Dataset* berukuran **35.066 sampel**, dengan performa yang sangat baik dan keseimbangan yang kuat antara kedua kelas.

**Model *machine learning* menggunakan algoritma `KNN`**

Parameter yang digunakan pada model yang menggunakan algoritma `KNN` adalah parameter default dari algoritma tersebut. Parameter yang digunakan adalah sebagai berikut:

- `n_neighbors`: Jumlah *neighbors* yang ditentukan. Nilai default dari parameter ini adalah `5`.
- `algorithm`: Algoritma pencarian *neighbors*. Nilai default dari parameter ini adalah `auto`.

```python
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

y_pred_train_knn = knn_model.predict(X_train)
y_pred_test_knn = knn_model.predict(X_test)

accuracy_train_knn = accuracy_score(y_pred_train_knn, y_train)
accuracy_test_knn = accuracy_score(y_pred_test_knn, y_test)

print('KNN - accuracy_train:', accuracy_train_knn)
print('KNN - accuracy_test:', accuracy_test_knn)
print('Classification Report:\n', classification_report(y_test, y_pred_test_knn))
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
KNN - accuracy_train: 0.9733641328371191
KNN - accuracy_test: 0.9609593338276393
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.96      0.96     17483
           1       0.96      0.96      0.96     17583

    accuracy                           0.96     35066
   macro avg       0.96      0.96      0.96     35066
weighted avg       0.96      0.96      0.96     35066
```

Model KNN menunjukkan performa yang sangat baik pada *dataset*.

- Akurasi:
  - Data latih: **97,34%**
  - Data uji: **96,10%**
  - Hasil ini menunjukkan model dapat menggeneralisasi dengan baik pada data uji.

- Data Klasifikasi:
  - **Kelas 0** (Tidak Diabetes):
    - Presisi: **96%**
    - Recall: **96%**
    - F1-Score: **96%**
  - **Kelas 1** (Diabetes):
    - Presisi: **96%**
    - Recall: **96%**
    - F1-Score: **96%**

- Rata-rata metrik untuk seluruh kelas:
  - Presisi: **96%**
  - Recall: **96%**
  - F1-Score: **96%**

- *Dataset* berukuran **35.066 sampel**, dengan keseimbangan performa yang sangat baik antara kedua kelas.

Dari hasil pelatihan dan evaluasi masing-masing model dengan ketiga algoritma yang telah disebutkan sebelumnya, dapat dilihat bahwa algoritma Random Forest memiliki rata-rata metrik paling tinggi dari kedua algoritma lainnya. Berdasarkan kelebihan-kelebihan dari algoritma tersebut dan hasil akurasi yang memadai serta merupakan nilai tertinggi dibanding algoritma lainnya, algoritma Random Forest menjadi algoritma yang cocok untuk dijadikan sebagai solusi.


## Evaluation
Berdasarkan hasil dari proses *modelling* yang dimana telah dipilih model terbaik yaitu model yang menggunakan algoritma *Random Forest*. Metrik evaluasi yang akan digunakan adalah sebagai berikut:

1. **Accuracy**  
   - Mengukur proporsi prediksi yang benar dari total data.  
   - Formula:  
      **Accuracy = (TP + TN) / (TP + TN + FP + FN)**  

      Informasi:
      - TP: True Positive (prediksi benar kelas positif)
      - TN: True Negative (prediksi benar kelas negatif)
      - FP: False Positive (prediksi salah kelas positif)
      - FN: False Negative (prediksi salah kelas negatif)

2. **Precision**  
   - Mengukur seberapa akurat model dalam memprediksi kelas positif dengan menunjukkan seberapa banyak prediksi positif yang benar dari semua prediksi positif.
   - Formula:  
      **Precision = TP / (TP + FP)** 

3. **Recall**  
   - Mengukur seberapa baik model mendeteksi kelas positif yang benar dengan menunjukkan seberapa banyak data positif yang benar-benar terdeteksi oleh model.
   - Formula:  
      **Recall = TP / (TP + FN)**  

4. **F1-Score**  
   - Rata-rata harmonis antara precision dan recall, memberikan keseimbangan antara keduanya. F1-Score digunakan ketika kita ingin keseimbangan antara precision dan recall.
   - Formula:  
      **F1-Score = 2 * (Precision * Recall) / (Precision + Recall)**  

5. **Support**  
   - Jumlah data sebenarnya untuk setiap kelas. Support memberikan informasi tentang jumlah sampel yang ada untuk masing-masing kelas dalam dataset.


```python
# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_rf)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest Model")
plt.show()
```

Kode diatas menghasilkan *output* sebagai berikut:

![Confusion matrix for Random Forest model](image-8.png)

```python
y_pred_test_rf = rf_model.predict(X_test)
accuracy_test_rf = accuracy_score(y_pred_test_rf, y_test)
print('Random Forest - accuracy_test:', accuracy_test_rf)
print('Classification Report:\n', classification_report(y_test, y_pred_test_rf))
```

Kode diatas menghasilkan *output* sebagai berikut:

```python
Random Forest - accuracy_test: 0.9785547253750071
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.99      0.98     17697
           1       0.99      0.97      0.98     17369

    accuracy                           0.98     35066
   macro avg       0.98      0.98      0.98     35066
weighted avg       0.98      0.98      0.98     35066
```

Berikut informasi yang didapatkan dari hasil *confusion matrix* dan *classification report*,

- **Akurasi**: **97.86%** pada data uji.
- **Precision**:
  - Kelas 0: **0.97**
  - Kelas 1: **0.99**
- **Recall**:
  - Kelas 0: **0.99**
  - Kelas 1: **0.97**
- **F1-score**:
  - Kelas 0 dan Kelas 1: **0.98**
- **Support**:
  - Kelas 0: **17,697 sampel**
  - Kelas 1: **17,369 sampel**
- **Jumlah total sampel**: **35,066 sampel**

Model ini mencapai tingkat **akurasi sebesar 97.86%** pada data uji, menunjukkan performa yang sangat baik. **Precision** untuk kelas 0 adalah **0.97** dan kelas 1 adalah **0.99**, sementara **recall** untuk kelas 0 mencapai **0.99** dan kelas 1 sebesar **0.97**. **F1-score** untuk kedua kelas adalah **0.98**, menunjukkan keseimbangan antara precision dan recall. Dengan total **35,066 sampel**, distribusi data cukup seimbang, yakni **17,697 untuk kelas 0** dan **17,369 untuk kelas 1**. Model ini mampu mengklasifikasikan kedua kelas dengan sangat baik.

## Referensi
[1]	L. Puspitasari, “Rumah sakit dengan pelayanan berkualitas,” Siloam Hospitals. Accessed: Nov. 25, 2024. [Online]. Available: https://www.siloamhospitals.com/informasi-siloam/artikel/angka-diabetes-di-indonesia-semakin-tinggi-berikut-faktanya-1
[2]	National Institute of Diabetes and Digestive and Kidney Disease, “What is diabetes?,” NIDDK - National Institute of Diabetes and Digestive and Kidney Diseases, Oct. 04, 2024. Accessed: Nov. 25, 2024. [Online]. Available: https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes
[3]	M. E. Febrian, F. X. Ferdinan, G. P. Sendani, K. M. Suryanigrum, and R. Yunanda, “Diabetes prediction using supervised machine learning,” Procedia Computer Science, vol. 216, pp. 21–30, 2023, doi: 10.1016/j.procs.2022.12.107.