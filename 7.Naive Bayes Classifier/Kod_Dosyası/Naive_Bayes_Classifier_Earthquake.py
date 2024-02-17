import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def plot_accuracy_wave_adjusted(accur):
    """
    Bu plot fonksiyonunda modelin doğruluğunu 5 aralıklarla çizdiriyoruz.
    """
    x_points = range(0, 10)
    y_points = [accur] * len(x_points)

    plt.plot(x_points, y_points, linestyle='--')
    plt.ylim(0, 100)  # Y eksenini 0 ile 100 arasında ayarlıyot
    plt.yticks(np.arange(0, 101, 5))  # Y yi 5 in katları şeklinde eksen değerleri
    plt.xlabel('Points')
    plt.ylabel('Başarı Oranı')
    plt.title('Deprem Başarı Oranı Modeli')
    plt.show()

# Veri setini yükle
VeriSet = pd.read_csv("C:/Users/Murathan/Documents/OKUL/3.sınıf/Zeki Sistemler/7.Naive Bayes Classifier/Veri_Seti/earthquake_1995-2023.csv")

# Gereksiz sütunları sil
del VeriSet["title"]
del VeriSet["location"]
del VeriSet["country"]
del VeriSet["continent"]

# Eksik değerleri işleme
VeriSet["alert"] = VeriSet["alert"].fillna("red")

# Tarih ve saat sütununun veri türünü değiştirme
VeriSet["date_time"] = pd.to_datetime(VeriSet["date_time"], dayfirst=True)
VeriSet["date_time"] = pd.DatetimeIndex(VeriSet["date_time"]).month

# Etiket kodlaması
le = LabelEncoder()
alert_le = LabelEncoder()
magtype_le = LabelEncoder()
net_le = LabelEncoder()
VeriSet["alert"] = alert_le.fit_transform(VeriSet["alert"])
VeriSet["magType"] = magtype_le.fit_transform(VeriSet["magType"])
VeriSet["net"] = net_le.fit_transform(VeriSet["net"])

# Histogram ve plot yani grafik tablolarını göster
VeriSet.hist()
plt.show()

# Veri kümesini dilimleme
Gercek = VeriSet.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
Test = VeriSet.iloc[:, [5]]

# Dengeleme tekniğini kullanarak verileri dengele
s = SMOTE()
Gercek_Veri, Test_Veri = s.fit_resample(Gercek, Test)

# Özellik Ölçeklendirme
ss = StandardScaler()
x_scaled = ss.fit_transform(Gercek_Veri)

# Modeli Eğitme %80 eğitim için %20 ise test verisi veri
# Veri setinin her zaman bölünme şeklinin aynı olması için randomstate 11 olarak kullanılır.
x_train, x_test, y_train, y_test = train_test_split(x_scaled, Test_Veri, random_state=11, test_size=0.2)

# Gaussian Naive Bayes Model ve eğitimi
nb = GaussianNB()
#bağımsız değişken x_train bağımlı değişken y_train
#np.ravel y_traini düzleştirir yanı uygun formatta olmasını sağlar.
nb.fit(x_train, np.ravel(y_train))

# Modelin performansını değerlendirme
#x_test üzerinde tahmin yapar
y_pred = nb.predict(x_test)
#bu satır tahminlerin doğruluğunu ölçer.
#modelin tahminlerini (y_pred), gerçek değerler (y_test) ile karşılaştırır ve doğru tahminlerin oranını hesaplar.
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Başarı oranı: {accuracy:.2f}%")

# Doğruluk grafiğini çizme
plot_accuracy_wave_adjusted(accuracy)

#kullanıcı seçimi
user_choice = input("Veri setinizin doğru olduğunu düşünüyorsanız 1'i, eğer yanlış olduğunu düşünüyorsanız 0'ı yazıp Enter tuşuna basınız: ")

if user_choice == "1":
    print("Görüşmek üzere...")
elif user_choice == "0":
    # Veri setini dataset'e aktar
    VeriSet.to_csv("kontrol_edilmis_veri_seti.csv", index=False)
    print("Veri seti başarıyla aktarıldı.")
    # Veriyi kaydet
    VeriSet.to_csv("veri_seti_yeni.csv", index=False)
    print("False sınıflandırılan veriler dataset'e eklendi ve veri_seti_yeni.csv olarak kaydedildi.")


    # Veri kümesini dilimleme
    Gercek = VeriSet.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    Test = VeriSet.iloc[:, [5]]

    # Dengeleme tekniğini kullanarak verileri dengele
    s = SMOTE()
    Gercek_Veri, Test_Veri = s.fit_resample(Gercek, Test)

    # Özellik Ölçeklendirme
    ss = StandardScaler()
    x_scaled = ss.fit_transform(Gercek_Veri)

    # Modeli Eğitme
    # Eğitim ve Test Verilerine Bölme
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, Test_Veri, random_state=11, test_size=0.2)

    # Gaussian Naive Bayes Model
    nb = GaussianNB()
    nb.fit(x_train, np.ravel(y_train))

    # Modelin performansını değerlendirme
    y_pred = nb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Başarı oranı: {accuracy:.2f}%")
    plot_accuracy_wave_adjusted(accuracy)
else:
    print("Geçersiz bir seçenek girdiniz.")