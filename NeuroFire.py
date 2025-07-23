import os
import time
import random
import webbrowser
from io import StringIO
import requests
import pandas as pd
import folium
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()
API_KEY = os.getenv("NASA_API_KEY")

# Sabitler
SOURCE = "VIIRS_SNPP_NRT"
COUNTRY = "TUR"
DAYS = "1"
REFRESH_RATE_SECONDS = 10
MODEL_PATH = "D:\\Software\\Projeler\\NeuroFire\\NeuroFire.h5"
KARAR_ESIGI = 0.85
DATASET_DIR = "D:\\Software\\Projeler\\NeuroFire\\fire_dataset"
HTML_FILE = "NeuroFire.html"

# Yapay Zekâ Modelini Yükle
print("Yapay zekâ modeli yükleniyor...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

# Simülasyon Görüntüleri Yükleniyor
print("Simülasyon için veri tabanı hazırlanıyor...")
try:
    fire_images_path = os.path.join(DATASET_DIR, "fire_images")
    non_fire_images_path = os.path.join(DATASET_DIR, "non_fire_images")

    yangin_resimler = [os.path.join(fire_images_path, f)
                       for f in os.listdir(fire_images_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    normal_resimler = [os.path.join(non_fire_images_path, f)
                       for f in os.listdir(non_fire_images_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"{len(yangin_resimler)} yangın, {len(normal_resimler)} normal resim yüklendi.")
except Exception as e:
    print(f"Veri seti hazırlanırken hata oluştu: {e}")
    exit()

# Tahmin Fonksiyonu
def tahmin_et_yangin(image_path):
    try:
        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction_non_fire = model.predict(img_array, verbose=0)[0][0]
        ihtimal_fire = 1 - prediction_non_fire
        if ihtimal_fire > KARAR_ESIGI:
            return "YANGIN", ihtimal_fire
        else:
            return "NORMAL", ihtimal_fire
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return "HATA", 0

# NASA API'den Veri Çekme
def get_fire_data():
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{API_KEY}/{SOURCE}/{COUNTRY}/{DAYS}"
    print(f"[{time.strftime('%H:%M:%S')}] NASA'dan güncel veri çekiliyor...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            print("Veri başarıyla alındı.")
            return response.text
        else:
            print(f"API Hatası: {response.status_code}")
            return None
    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return None

# Harita Oluşturma
def create_map(fire_data_text):
    if fire_data_text is None or len(fire_data_text.strip().splitlines()) <= 1:
        print("Gösterilecek yangın verisi bulunamadı.")
        return

    try:
        df = pd.read_csv(StringIO(fire_data_text))
        if not all(col in df.columns for col in ['latitude', 'longitude', 'acq_date', 'acq_time', 'confidence']):
            print("Veri formatı eksik.")
            return

        df = df[df['confidence'] != 'l']  # Düşük güven filtrelendi
        print(f"{len(df)} yüksek güvenilirlikli nokta haritaya işleniyor...")

        turkiye_map = folium.Map(location=[39.925533, 32.866287], zoom_start=6)

        # Otomatik yenileme ekle
        refresh_html = f"""
        <meta http-equiv="refresh" content="{REFRESH_RATE_SECONDS}">
        """
        turkiye_map.get_root().html.add_child(folium.Element(refresh_html))

        for _, nokta in df.iterrows():
            enlem = nokta['latitude']
            boylam = nokta['longitude']
            tarih = nokta['acq_date']
            saat = nokta['acq_time']

            # Simülasyon için rastgele bir görsel
            is_fire_simulation = random.choice([True, False, False, False, False])
            test_image = random.choice(yangin_resimler if is_fire_simulation else normal_resimler)
            sonuc, ihtimal = tahmin_et_yangin(test_image)

            if sonuc == "YANGIN":
                icon = folium.Icon(color='red', icon='fire', prefix='fa')
                popup_text = f"""
                <h4><b>🔥 DOĞRULANMIŞ YANGIN</h4></b>
                <b>Yangın Olasılığı:</b> <font color="red">{ihtimal:.2%}</font><br>
                <b>Tarih:</b> {tarih}<br>
                <b>Konum:</b> {enlem:.4f}, {boylam:.4f}
                """
            else:
                icon = folium.Icon(color='orange', icon='info-circle', prefix='fa')
                popup_text = f"""
                <h4><b>🚨 TERMAL ANOMALİ</h4></b>
                <b>Yangın Olasılığı:</b> {ihtimal:.2%}<br>
                <b>Tarih:</b> {tarih}<br>
                <b>Konum:</b> {enlem:.4f}, {boylam:.4f}
                """

            folium.Marker(
                location=[enlem, boylam],
                icon=icon,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(turkiye_map)

        turkiye_map.save(HTML_FILE)
        print(f"[{time.strftime('%H:%M:%S')}] Harita '{HTML_FILE}' olarak güncellendi.")

    except Exception as e:
        print(f"Harita oluşturulurken hata oluştu: {e}")

# Ana Döngü
if __name__ == "__main__":
    print("NeuroFire Final Paneli Başlatılıyor...")
    initial_data = get_fire_data()
    create_map(initial_data)

    if os.path.exists(HTML_FILE):
        webbrowser.open(HTML_FILE)

    try:
        while True:
            print(f"\nSonraki güncelleme {REFRESH_RATE_SECONDS} saniye sonra...")
            time.sleep(REFRESH_RATE_SECONDS)
            data = get_fire_data()
            create_map(data)
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu. Çıkılıyor...")
