import pandas as pd
import folium as fl

print("Yangın Verisi Okunuyor...")

try:
    df = pd.read_csv("ham_yangin_verisi.csv")
    print(f"{len(df)} adet potansiyel yangın noktası bulundu.")
except FileNotFoundError:
    print("Hata: ham_yangin_verisi.csv dosyası bulunamadı!")
    print("Lütfen önce veriCekici.py scriptini çalıştırarak veriyi indirin.")
    exit()

print("Harita Oluşturuluyor...")

turkiyeHaritasi = fl.Map(location=[39.925533, 32.7866287], zoom_start=6)

for index, nokta in df.iterrows():
    enlem = nokta["latitude"]
    boylam = nokta["longitude"]
    tarih = nokta["acq_date"]
    saat = nokta["acq_time"]
    guven = nokta["confidence"]

    popup_metni = f"""<b>Tespit Tarihi:</b> {tarih}<br>
                      <b>Tespit Saati:</b> {saat}<br>
                      <b>Güven Seviyesi:</b> {guven}"""

    fl.CircleMarker(
        location=[enlem, boylam],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.7,
        popup=fl.Popup(popup_metni, max_width=300)
    ).add_to(turkiyeHaritasi)

# 🔄 Harita dosyasını döngü dışında tek seferde kaydet:
turkiyeHaritasi.save("NeuroFire.html")
print("\nHarita başarıyla oluşturuldu!")
print("NeuroFire.html dosyasını açarak güncel yangın haritasını görüntüleyebilirsiniz.")
