import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "NeuroFire.h5"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
KARAR_ESIGI = 0.90

print("Yapay Zeka Modeli Yükleniyor....")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model Başarıyla Yüklendi!")
except Exception as e:
    print(f"Model Yüklenirken Şöyle Bir Hata Oluştu: {e}")
    exit()

# Örnek: Burada sınıf indekslerini manuel belirt, 
# ancak asıl doğru yolu model eğitirken kullanılan sınıf indekslerini öğrenmek:
class_indices = {'fire': 0, 'non_fire': 1}  # Bu değeri kendi verinize göre değiştirin!
print(f"Sınıf indeksleri: {class_indices}")

def tahminEt(image_path):
    try:
        img = image.load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        image_array = image.img_to_array(img)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0  # Normalizasyon

        prediction = model.predict(image_array)[0][0]
        


        if prediction < KARAR_ESIGI:
            ihtimal = 1 - prediction
            print(f"SONUÇ : YANGIN TESPİT EDİLDİ! %{KARAR_ESIGI:.0%}")
        else:
            ihtimal = prediction
            print(f"SONUÇ : YANGIN TESPİT EDİLEMEDİ! %{KARAR_ESIGI:.0%}")
            
    except FileNotFoundError:
        print(f"Hata: '{image_path}' adında bir dosya bulunamadı!")
    except Exception as e:
        print(f"Tahmin sırasında bir hata oluştu: {e}")

if __name__ == "__main__":
    print("\n--- TEST 1: YANGIN GÖRÜNTÜSÜ ---")
    tahminEt("test_yangin.jpg")

    print("\n--- TEST 2: NORMAL GÖRÜNTÜ ---")
    tahminEt("test_normal.jpg")
