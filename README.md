<div align="center">

# 🏭 PredictWise

**AI destekli, gerçek zamanlı çalışan kestirimci bakım karar destek sistemi**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-6E56CF?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=flat&logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/status-active--development-brightgreen?style=flat)

</div>

---

## 📌 Proje Hakkında

PredictWise, endüstriyel makinelerin sensör verilerinden (sıcaklık, devir hızı, tork, takım aşınması) yola çıkarak **arıza riskini önceden tahmin eden** bir karar destek sistemi. [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) üzerinde eğitilmiş bir Random Forest modeli kullanır ve her makine için:

- **Risk Skoru** (0–100)
- **Sağlık Skoru**
- **Durum** (Healthy / Warning / Critical)
- **Somut bir bakım önerisi**

üretir.

Proje, statik bir veri setinin üzerine kurulu bir analiz aracı olmaktan çıkıp, **kendi ürettiği simüle edilmiş canlı sensör akışını** işleyen gerçek zamanlı bir sisteme dönüştürüldü — bkz. [Canlı Veri Mimarisi](#-canlı-veri-mimarisi).

> **Not:** Sistem gerçek fabrika sensörlerine değil, gerçekçi bir simülasyona bağlıdır. Mimari, gerçek sensör verisine geçişte sadece veri kaynağı katmanı değiştirilerek uyarlanabilecek şekilde tasarlandı.

---

## 🖼️ Ekran Görüntüleri

> _[Buraya Dashboard, Explainability ve Model Performance sayfalarından ekran görüntüsü eklenecek]_

---

## ✨ Özellikler

| Sayfa | Ne İşe Yarar |
|---|---|
| 📊 **Dashboard** | Canlı filo özeti, AI destekli durum özeti, KPI'lar, bakım öncelik paneli |
| 🔮 **AI Prediction** | Tekil makine girişi veya toplu CSV yükleme ile anlık tahmin |
| 📈 **Analytics** | Filo geneli dağılımlar, korelasyon analizi, makine tipine göre risk kırılımı |
| 🧠 **Explainability** | SHAP tabanlı global ve lokal açıklamalar — model neden bu kararı verdi? |
| 📉 **Model Performance** | Accuracy, precision/recall, ROC eğrisi, confusion matrix — **gerçek etiket bulunamazsa sahte metrik göstermez** |
| ⚙️ **Settings** | Oturum tercihleri, canlı model bilgisi, risk eşikleri (kod üzerinden canlı okunur), otomatik yenileme |
| ℹ️ **About** | Proje tanıtımı, mimari, teknoloji yığını, canlı istatistikler |

---

## 🔴 Canlı Veri Mimarisi

Dashboard artık iki modda çalışabilir:

- **LIVE** — simülatör ve worker çalışıyorsa, filodaki her makinenin en güncel skorlanmış durumu gösterilir, sayfa kendi kendine yenilenir.
- **SNAPSHOT (historical)** — canlı veri yoksa, sistem çökmek yerine orijinal tarihsel veri setine dürüstçe geri düşer.

```
simulator/  →  sensör okuması üretir (30 sanal makine, gerçekçi tool-wear birikimi)
      │
      ▼
data/live_readings.db  (SQLite, ham okumalar)
      │
      ▼
worker/    →  src/pipeline.py'yi DEĞİŞTİRMEDEN kullanarak skorlar
      │
      ▼
data/live_readings.db  (scored_readings tablosu)
      │
      ▼
dashboard/ →  st.fragment ile otomatik yenilenen canlı görünüm
```

Skorlama mantığı (`src/pipeline.py`) veri kaynağından tamamen bağımsız yazıldığı için, simülatör ve gerçek/tarihsel veri **aynı pipeline'dan** geçer — iki ayrı kod yolu yoktur.

---

## 🛠️ Teknoloji Yığını

- **Python** — ana dil
- **Streamlit** — interaktif web arayüzü
- **scikit-learn** — Random Forest modeli
- **Pandas / NumPy** — veri işleme
- **Plotly** — interaktif grafikler
- **SHAP** — model açıklanabilirliği
- **SQLite** — canlı veri akışı için hafif veritabanı

---

## 🚀 Kurulum ve Çalıştırma

```bash
git clone https://github.com/<kullanici-adi>/predictwise.git
cd predictwise
pip install -r requirements.txt
```

### Sadece dashboard'u (tarihsel veriyle) çalıştırmak için

```bash
streamlit run dashboard/app.py
```

### Canlı veri akışını da görmek için (3 ayrı terminal)

```bash
# Terminal 1 — sahte sensör verisi üretir
python simulator/generate_readings.py

# Terminal 2 — üretilen veriyi arka planda skorlar
python worker/score_worker.py

# Terminal 3 — dashboard
streamlit run dashboard/app.py
```

---

## 📁 Proje Yapısı

```
predictwise/
│
├── dashboard/
│   ├── app.py              # Routing — tek giriş noktası
│   ├── theme.py             # Tek merkezi tasarım sistemi (renk, kart, grid)
│   ├── layout.py            # Paylaşılan section()/card_grid() bileşenleri
│   ├── data.py               # Tarihsel + canlı veri erişimi
│   ├── preferences.py        # Oturum tercihleri
│   └── pages/
│       ├── dashboard.py
│       ├── prediction.py
│       ├── analytics.py
│       ├── explainability.py
│       ├── model_performance.py
│       ├── settings.py
│       └── about.py
│
├── src/
│   ├── pipeline.py           # feature engineering → preprocessing → prediction → business rules
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── business_rules.py
│   └── explainability.py
│
├── simulator/                # Canlı sensör verisi üretici
├── worker/                   # Arka planda skorlama servisi
├── models/                   # Eğitilmiş Random Forest (random_forest.pkl)
├── data/                     # AI4I 2020 veri seti + canlı SQLite veritabanı
└── .streamlit/config.toml
```

---

## 🗺️ Yol Haritası

- [x] 7 sayfalık dashboard
- [x] Simüle edilmiş canlı veri akışı (simulator + worker)
- [x] Otomatik yenileme
- [ ] Zaman boyutlu trend grafikleri (makine bazlı risk geçmişi)
- [ ] Kritik durum bildirimleri
- [ ] Docker ile tek komutla ayağa kalkan sistem

---

## 👤 Geliştiren

**Nurefşan** — Bilgisayar Mühendisliği & Endüstri Mühendisliği (Çift Anadal), KTO Karatay Üniversitesi

---

## 📄 Lisans

_[Lisans seçimi eklenecek — MIT önerilir]_
