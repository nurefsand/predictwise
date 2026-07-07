# SmartWatch for Machines: Predictive Maintenance

## Problem: Endüstriyel Kayıp
Üretim hatlarında beklenmedik makine arızaları, hem yüksek bakım maliyetlerine hem de planlanmamış duruş sürelerine (downtime) yol açmaktadır. Geleneksel "arıza sonrası bakım" yöntemi, modern endüstriyel standartlarda verimsizdir.

## Çözüm: Proaktif Kestirimci Bakım
Bu proje, makine sensör verilerini kullanarak arızaları önceden tahmin eden ve operatörler için bir "erken uyarı sistemi" olarak işlev gören bir modelleme çalışmasıdır.

### Temel Analitik Yaklaşım
* **Veri Keşfi:** Makine tipleri ve arıza türleri (HDF, PWF, OSF, TWF) arasındaki korelasyon analiz edildi.
* **Feature Engineering:** Fiziksel kurallardan hareketle `temp_diff` (ısı verimliliği) ve `power` (mekanik yük) gibi modelin tahmin gücünü artıran yeni öznitelikler türetildi.
* **İş Değeri:** Arıza olasılığını %80 ve üzeri olarak tahmin eden bir risk skorlama mekanizması tasarlandı.

## Teknik Detaylar
* **Dataset:** AI4I 2020 Predictive Maintenance Dataset.
* **Analiz:** Korelasyon matrisleri ile sensörler arası bağımlılıklar incelendi.
* **Teknoloji:** Python, Pandas, Seaborn, Scikit-learn (Devam ediyor).

## Hedeflenen Çıktılar
Bu proje sonucunda;
1. Kritik arıza türlerinin (HDF, PWF) erken tespiti.
2. Bakım maliyetlerini minimize edecek risk bazlı bir dashboard prototipi oluşturulması.
3. Arıza anomali tespiti için savunulabilir bir veri modeli geliştirilmesi.