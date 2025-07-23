# 🔥 NeuroFire – Yapay Zekâ Destekli Yangın Tespit Sistemi

**NeuroFire**, uydu verileri ve görüntü işleme teknikleri ile çalışan, yapay zekâ destekli bir **yangın tespit ve izleme sistemidir**. Sistem, NASA FIRMS API'sinden alınan verilerle Türkiye üzerindeki termal anomalileri analiz eder, rastgele görseller üzerinden tahmin yapar ve etkileşimli bir harita oluşturur.

---

## 🎯 Amaç

- Orman yangınlarını erken tespit etmek
- Termal uydu verilerini analiz etmek
- Görsel sınıflandırma ile yangın olasılığı belirlemek
- Türkiye haritası üzerinde potansiyel yangın noktalarını göstermek

---

## 🧠 Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- Folium
- Pandas, NumPy
- NASA FIRMS API

---

## 📂 Proje Dosya Yapısı

| Dosya / Klasör           | Açıklama                                                  |
|--------------------------|------------------------------------------------------------|
| `NeuroFire.py`           | Ana uygulama dosyası                                       |
| `modelEgitici.py`        | Yapay zekâ modelini eğitir                                |
| `tahminEt.py`            | Görselleri analiz edip yangın olasılığı tahmin eder       |
| `harita_olusturucu.py`   | Türkiye haritası üzerinde işaretleme işlemini yapar       |
| `egitim_grafigi.png`     | Model eğitim sürecinin görsel grafiği (isteğe bağlı)       |

---

## ⚙️ Kurulum ve Kullanım

```bash
# Sanal ortam oluştur (isteğe bağlı ama önerilir)
python -m venv yangin_env
yangin_env\Scripts\activate  # Windows

# Gereken kütüphaneleri yükle
pip install -r requirements.txt
